import torch
from torch.utils.data import Dataset   
from gpt2.tokenizer import BytePairTokenizer


class GPT2Dataset(Dataset):
    """Dataset class for preparing tokenized text data for GPT-2 training.

        This class handles:
           - Loading raw text data from files
           - Tokenize text using tokenizer(Byte-pair Encoding)
           - Creating input-target pair based on context length
           - Generating batches for training and evaluation
        
        Attributes:
            - tokenizer : An Instance of tokenizer class which tokenize processed text
            - txt : text dataset
            - max_length: The maximum training length for each training example.
            - stride: the number
        
        Methods:
            __len__(self)->int:
                Get Length of tokens
            
            __getitem__(self,idx:int)->tuple[torch.Tensor,torch.Tensor]:
                Get input id and target id based on index

    """

    def __init__(self,text:str,tokenizer:BytePairTokenizer,max_length:int,stride:int) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        if (len(token_ids) < max_length):
            raise ValueError("Text length too short for given max_length")
        
        for i in range(0,len(token_ids)-max_length,stride):
            input_id = token_ids[i:i+max_length]
            target_id = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_id,dtype = torch.long))
            self.target_ids.append(torch.tensor(target_id,dtype =torch.long))
    

    def __len__(self)->int:
        """Get length of input ids
        """
        return len(self.input_ids)
    
    
    def __getitem__(self,idx:int)->tuple[torch.Tensor,torch.Tensor]:
        """Return input and target token IDs for a given index.

        Args:
            idx (int): Index of the sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - input_ids: Tensor of shape (max_length,)
                - target_ids: Tensor of shape (max_length,)
        """
        return self.input_ids[idx], self.target_ids[idx]
