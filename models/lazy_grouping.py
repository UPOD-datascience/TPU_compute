import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Dict, List, Any, Iterator, Union, Optional
import torch_xla.core.xla_model as xm

class LazyGroupingDataset(IterableDataset):
    """
    A wrapper dataset that applies grouping lazily during iteration.
    This allows for on-the-fly text grouping during training rather than preprocessing the entire dataset upfront.

    For streaming datasets, this is particularly beneficial as it:
    1. Reduces startup time dramatically
    2. Processes only the data needed for each batch
    3. Maintains the streaming nature of the dataset
    """

    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        max_seq_length: int,
        pad_token: int = 0,
        batch_size: int = 8,
        streaming: bool = True
    ):
        """
        Initialize the lazy grouping dataset wrapper.

        Args:
            dataset: The underlying dataset containing tokenized examples
            max_seq_length: Maximum sequence length for grouped examples
            pad_token: Token ID to use for padding (default: 0)
            batch_size: Batch size for grouping
            streaming: Whether the dataset is streaming (iterable) or not
        """
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.batch_size = batch_size
        self.streaming = streaming

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through the dataset, applying grouping on-the-fly.

        Returns:
            Iterator yielding batches of grouped examples
        """
        # Create an iterator from the underlying dataset
        dataset_iter = iter(self.dataset)

        # Keep yielding batches until we're done
        while True:
            try:
                # Get a batch of examples
                batch = []
                for _ in range(self.batch_size):
                    try:
                        item = next(dataset_iter)
                        batch.append(item)
                    except StopIteration:
                        if not batch:  # If batch is empty, we're truly done
                            raise
                        break  # Otherwise process the partial batch

                if not batch:
                    break

                # Convert list of dicts to dict of lists
                batch_dict = {k: [item[k] for item in batch] for k in batch[0].keys()}

                # Apply grouping to this batch
                grouped_batch = self._group_texts(batch_dict)

                # Yield each example in the grouped batch
                for i in range(len(grouped_batch[list(grouped_batch.keys())[0]])):
                    # Return the example without converting to tensor
                    # Let the data collator handle the tensor conversion
                    example = {k: grouped_batch[k][i] for k in grouped_batch.keys()}
                    yield example

            except StopIteration:
                # No more data
                break

    def _group_texts(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Group already tokenized texts into chunks of max_seq_length while respecting sample boundaries.

        Args:
            examples: Dictionary with keys like 'input_ids', 'attention_mask', etc. where each value
                     is a list of tokenized examples

        Returns:
            Dictionary with same keys but values chunked to max_seq_length with padding
        """
        result = {k: [] for k in examples.keys()}

        # Get a sample key to determine the number of examples
        sample_key = list(examples.keys())[0]

        # Loop through each tokenized example
        for i in range(len(examples[sample_key])):
            # Extract the current tokenized example for each feature
            current_example = {k: examples[k][i] for k in examples.keys()}

            # Calculate how many chunks we need for this example
            example_length = len(current_example[sample_key])
            num_chunks = (example_length + self.max_seq_length - 1) // self.max_seq_length  # Ceiling division

            # Split each feature into chunks
            for k, tokens in current_example.items():
                # Create chunks of max_seq_length
                chunks = []
                for j in range(0, example_length, self.max_seq_length):
                    chunk = tokens[j:min(j + self.max_seq_length, example_length)]

                    # Pad if necessary
                    if len(chunk) < self.max_seq_length:
                        chunk = chunk + [self.pad_token] * (self.max_seq_length - len(chunk))

                    chunks.append(chunk)

                # If we don't have enough chunks (unlikely but possible with different length features)
                while len(chunks) < num_chunks:
                    chunks.append([self.pad_token] * self.max_seq_length)

                # Add the chunks to the result
                result[k].extend(chunks)

        return result

    def __len__(self) -> Optional[int]:
        """
        Return the length of the dataset if available.
        For streaming datasets, this may not be available.

        Returns:
            Length of the dataset if available, else None
        """
        if not self.streaming and hasattr(self.dataset, "__len__"):
            # This is a very rough estimate as grouping will change the actual length
            return len(self.dataset) * self.max_seq_length
        return None
