"""Regression tests for streaming PPPL validation."""

import random
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, IterableDataset

from models.cpt_deberta_gpu import (
    FractionalPPPLDataset,
    evaluate,
    limit_validation_documents,
)


class StreamingExamples(IterableDataset):
    def __iter__(self):
        yield {"input_ids": [1, 10, 11, 0], "attention_mask": [1, 1, 1, 0]}
        yield {"input_ids": [1, 12, 13, 14], "attention_mask": [1, 1, 1, 1]}


class FakeDataset(list):
    def shuffle(self, seed, buffer_size=None):
        del buffer_size
        return FakeDataset(random.Random(seed).sample(self, len(self)))

    def select(self, indices):
        return FakeDataset(self[index] for index in indices)

    def take(self, count):
        return FakeDataset(self[:count])


class FakeModel:
    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, input_ids, attention_mask, labels):
        # Different batch losses expose incorrect averaging of the final short batch.
        return SimpleNamespace(loss=labels.ne(-100).sum().float())


class PPPLTests(unittest.TestCase):
    def test_validation_document_limit_is_reproducible(self):
        dataset = FakeDataset(range(20))
        args = SimpleNamespace(
            max_validation_documents=5,
            streaming_data=False,
            seed=17,
            shuffle_buffer_size=10,
        )

        first = limit_validation_documents(dataset, args)
        second = limit_validation_documents(dataset, args)

        self.assertEqual(len(first), 5)
        self.assertEqual(first, second)
        self.assertTrue(set(first).issubset(set(range(20))))

    def test_zero_validation_document_limit_keeps_all_documents(self):
        dataset = FakeDataset(range(4))
        args = SimpleNamespace(
            max_validation_documents=0,
            streaming_data=False,
            seed=17,
            shuffle_buffer_size=10,
        )

        self.assertIs(limit_validation_documents(dataset, args), dataset)

    def test_streaming_validation_document_limit_uses_take(self):
        dataset = FakeDataset(range(20))
        args = SimpleNamespace(
            max_validation_documents=5,
            streaming_data=True,
            seed=17,
            shuffle_buffer_size=3,
        )

        selected = limit_validation_documents(dataset, args)

        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, limit_validation_documents(dataset, args))

    def test_streaming_dataset_can_be_evaluated_repeatedly(self):
        tokenizer = SimpleNamespace(mask_token_id=99, all_special_ids=[0, 1, 99])
        dataset = FractionalPPPLDataset(StreamingExamples(), tokenizer, fraction=1)
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=lambda examples: {
                key: torch.tensor([example[key] for example in examples])
                for key in examples[0]
            },
        )

        first_pass = list(loader)
        second_pass = list(loader)
        self.assertEqual(len(first_pass), 3)  # Five targets, one per masked example.
        for first, second in zip(first_pass, second_pass):
            self.assertTrue(torch.equal(first["input_ids"], second["input_ids"]))
            self.assertTrue(torch.equal(first["labels"], second["labels"]))
            self.assertTrue(first["labels"].ne(-100).sum(dim=1).eq(1).all())
            self.assertTrue(first["input_ids"].eq(99).sum(dim=1).eq(1).all())

        self.assertTrue(first_pass[0]["attention_mask"][:, -1].eq(0).all())
        ppl, loss = evaluate(FakeModel(), loader, torch.device("cpu"))
        self.assertEqual(loss, 1.8)  # (2 * 2 + 2 * 2 + 1 * 1) / 5

    def test_no_eligible_targets_raises_clear_error(self):
        tokenizer = SimpleNamespace(mask_token_id=99, all_special_ids=[0, 1, 99])
        dataset = FractionalPPPLDataset(
            [{"input_ids": [1, 0], "attention_mask": [1, 0]}], tokenizer
        )
        with self.assertRaisesRegex(ValueError, "no eligible tokens"):
            evaluate(FakeModel(), DataLoader(dataset), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
