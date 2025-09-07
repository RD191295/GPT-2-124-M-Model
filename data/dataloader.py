from gpt2.tokenizer import BytePairTokenizer
from data.dataset import GPT2Dataset
from torch.utils.data import DataLoader   
import torch
from typing import Tuple

dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]] 

def create_data_loader(text:str,
                       batch_size:int = 4,
                       max_length:int = 256,
                       stride:int = 128,
                       shuffle:bool =True,
                       drop_last:bool = True,
                       num_workers:int = 0,
                       special_tokens:set[str]= {"<|endoftext|>"}
                       )->DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
        """Create a PyTorch DataLoader for GPT-2 training or evaluation.

        This function initializes a BytePairTokenizer, prepares the dataset
        using `GPT2Dataset`, and wraps it in a `torch.utils.data.DataLoader`.

        Args:
            text (str): Input text corpus to tokenize and process.
            batch_size (int, optional): Number of sequences per batch. Default is 4.
            max_length (int, optional): Maximum sequence length for input examples. Default is 256.
            stride (int, optional): Step size for sliding window across tokens. Default is 128.
            shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
            drop_last (bool, optional): Whether to drop the last incomplete batch. Default is True.
            num_workers (int, optional): Number of subprocesses for data loading. Default is 0.
            special_tokens (set[str], optional): Special tokens to initialize the tokenizer with.
                Default is {"<|endoftext|>"}.

        Returns:
            DataLoader: A PyTorch DataLoader yielding batches of
            `(input_ids, target_ids)` tensors where:
                - input_ids: Tensor of shape `(batch_size, max_length)`
                - target_ids: Tensor of shape `(batch_size, max_length)`
        """
        if not text:
                raise ValueError("Text should not be empty string")
        if batch_size <= 0:
                raise ValueError("Batch size should not be less than zero")
        if max_length <= 0:
            raise ValueError("max length should not be less than equal to zero")
        if stride <= 0:
            raise ValueError("stride should not be less than equal to zero")
        if stride > max_length:
            raise ValueError("stride should be less than max length")
        
        tokenizer = BytePairTokenizer(special_token = special_tokens)

        dataset = GPT2Dataset(text,tokenizer,max_length,stride)

        dataloader = DataLoader( # type: ignore
            dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = num_workers
        )


        return dataloader # type: ignore