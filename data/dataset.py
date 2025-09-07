import torch
from torch.utils.data import Dataset   
from gpt2.tokenizer import BytePairTokenizer
from typing import Tuple


class GPT2Dataset(Dataset):
    """Memory-efficient PyTorch Dataset for GPT-2 tokenized text.

        This class handles:
            - Takes a raw text string
            - Tokenizes text using Byte-Pair Encoding
            - Dynamically generates input-target pairs based on context length and stride

        Attributes:
            token_ids (list[int]): List of token IDs obtained from the tokenizer
            max_length (int): Number of tokens per input sequence
            stride (int): Step size to slide the window
            shift (int): Number of tokens to shift for the target sequence
    """
    def __init__(self, text: str, tokenizer: BytePairTokenizer,
                 max_length: int, stride: int, shift: int = 1) -> None:
        if not text:
            raise ValueError("text can not be empty")
        if shift < 1:
            raise ValueError("Shift cannot be less than 1")
            
        self.token_ids = tokenizer.encode(text)

        self.max_length = max_length
        self.stride = stride
        self.shift = shift
        
        if len(self.token_ids) < max_length + shift:
            raise ValueError("Text length too short for given max_length + shift")

        # Calculate no of sample
        self.num_samples = ((len(self.token_ids)- max_length) // stride ) + 1

            
    def __len__(self) -> int:
        """Return number of input-target pairs in dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return input and target token IDs for a given index.

        Args:
            idx (int): Index of the sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - input_ids: Tensor of shape (max_length,)
                - target_ids: Tensor of shape (max_length,)
        """
        start = idx * self.stride
        end = start + self.max_length
        if end+self.shift > len(self.token_ids):
            # Handle edge case for last few tokens
            end = len(self.token_ids) - self.shift
            start = max(0, end - self.max_length)
        
        inputs = torch.tensor(self.token_ids[start:end], dtype=torch.long)
        targets = torch.tensor(self.token_ids[start+self.shift:end+self.shift], dtype=torch.long)
        return inputs, targets
