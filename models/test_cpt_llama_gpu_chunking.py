"""Regression tests for optional sentence-aware Llama chunking."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models.cpt_llama_gpu import _sentence_token_boundaries, group_texts, tokenize_function


class FakeSegmenter:
    def __init__(self, sentences):
        self.sentences = sentences

    def segment(self, text):
        end = 0
        spans = []
        for sentence in self.sentences:
            end += len(sentence)
            spans.append(SimpleNamespace(end=end, sent=sentence))
        return spans


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, texts, **kwargs):
        assert kwargs["return_offsets_mapping"]
        return {
            "input_ids": [[1] + list(range(10, 10 + len(text))) for text in texts],
            "attention_mask": [[1] * (len(text) + 1) for text in texts],
            "offset_mapping": [[(0, 0)] + [(i, i + 1) for i in range(len(text))] for text in texts],
        }


class SentenceChunkingTests(unittest.TestCase):
    def setUp(self):
        import models.cpt_llama_gpu as module
        module._debug_tokenization = False
        module._debug_group_texts = False

    def test_sentence_and_paragraph_boundaries_preserve_tokens_and_eos(self):
        text = "One. Two.\n\nLast."
        tokenizer = FakeTokenizer()
        with patch("models.cpt_llama_gpu.pysbd.Segmenter", return_value=FakeSegmenter(["One. ", "Two.\n\n", "Last."])):
            tokenized = tokenize_function({"text": [text]}, tokenizer, sentence_aware=True)

        self.assertNotIn("offset_mapping", tokenized)
        self.assertEqual(tokenized["sentence_boundaries"][0], [6, 12, len(text) + 1])
        chunks = group_texts(tokenized, max_seq_length=8, pad_token=0)
        actual = [token for ids, mask in zip(chunks["input_ids"], chunks["attention_mask"])
                  for token, present in zip(ids, mask) if present]
        self.assertEqual(actual, tokenized["input_ids"][0])
        self.assertEqual(actual.count(99), 1)
        self.assertEqual(len(chunks["input_ids"]), 3)
        self.assertTrue(all(len(chunk) == 8 for chunk in chunks["input_ids"]))

    def test_uses_pysbd_spans_without_rematching_sentence_strings(self):
        text = "One. Two."
        offsets = [(0, 0)] + [(i, i + 1) for i in range(len(text))]
        segmenter = SimpleNamespace(segment=lambda _: [SimpleNamespace(sent="not an exact substring", end=5)])
        self.assertEqual(_sentence_token_boundaries(text, offsets, segmenter), [6])

    def test_paragraph_boundary_even_without_sentence_boundary(self):
        text = "First.\n\nSecond."
        offsets = [(0, 0)] + [(i, i + 1) for i in range(len(text))]
        self.assertIn(9, _sentence_token_boundaries(text, offsets, FakeSegmenter([text])))

    def test_oversized_sentence_falls_back_to_hard_split(self):
        tokens = list(range(10)) + [99]
        chunks = group_texts({
            "input_ids": [tokens],
            "attention_mask": [[1] * len(tokens)],
            "sentence_boundaries": [[10]],
        }, max_seq_length=4, pad_token=0)
        self.assertEqual([ids[:sum(mask)] for ids, mask in zip(chunks["input_ids"], chunks["attention_mask"])],
                         [tokens[:4], tokens[4:8], tokens[8:]])

    def test_default_fixed_length_behavior_unchanged(self):
        tokens = list(range(7))
        chunks = group_texts({"input_ids": [tokens], "attention_mask": [[1] * 7]},
                             max_seq_length=4, pad_token=0)
        self.assertEqual(chunks["input_ids"], [[0, 1, 2, 3], [4, 5, 6, 0]])
        self.assertEqual(chunks["attention_mask"], [[1, 1, 1, 1], [1, 1, 1, 0]])


if __name__ == "__main__":
    unittest.main()
