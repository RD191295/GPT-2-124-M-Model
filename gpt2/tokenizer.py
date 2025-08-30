import tiktoken

class BytePairTokenizer:
    """Byte Pair Tokenizer with Encode and Decode

        This class Provide method to encode text into IDS and decode
        token IDs back to text. It support custom special token
        (e.g. <|endoftext|>) and interegrate with 'tiktoken' gpt-2 
        tokenizer

        Attributes:
             tokenizer: The GPT2 Tokenizer Instance from tiktoken
             special_token (set[str]): set of special token to be recognise during encoding.
        
        Methods:
            encode(self,text:str) ->list[int]:
                convert input text into list of token IDS

            decode(self,ids:list[int])->str:
                convert list of token IDs into text
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