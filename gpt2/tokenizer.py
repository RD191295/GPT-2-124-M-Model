import tiktoken

class BytePairTokenizer:
    """Byte Pair Tokenizer with Encode and Decode
    """
    def __init__(self, special_token: set[str] = {"<|endoftext|>"}) -> None:
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.special_token = special_token

    def encode(self,text:str) ->list[int]:
        """Encode text into token IDs.

        Args:
            text (str): Input text to tokenize.

        Returns:
            list[int]: Token IDs.
        """
        ids = []

        if text:
            ids = self.tokenizer.encode(text,allowed_special=self.special_token)
   
        return ids
    
    
    def decode(self,ids:list[int])->str:
        """Decode token ids to text.

        Args:
            ids (list[int]): Token IDs.

        Returns:
            str: Decoded string
        """
         
        text = self.tokenizer.decode(ids)

        return text