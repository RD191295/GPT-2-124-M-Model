import json
from json import JSONDecodeError


class ConfigObject(object):
    def __init__(self):
        pass

    pass

class Config:

    def __init__(self,Config_path):
        """
            Init COnstructuor
            In:
            Config_path : COnfig File Path
        """
        super().__init__()
        self.Config_path = Config_path
        self.config = ConfigObject()

    def load(self):
        """
            File Load method
        """
        try:
            with open(self.Config_path,'r') as f:
                configs = json.load(f)

                for key, val in configs.items():
                    self.config.__setattr__(key,val)

            required_keys = ["vocab_size", "context_length", "emb_dim", "n_heads", "n_layers", "drop_rate"]

            for key in required_keys:
                if not hasattr(self.config, key):
                    raise ValueError(f"Config missing required key:{key}")
            
            if not isinstance(self.config.vocab_size , int) or self.config.vocab_size <= 0:
                raise ValueError(f"vocab size smust be positive value")
            
            if not isinstance(self.config.context_length , int) or self.config.context_length <= 0:
                raise ValueError(f"context length must be positive value")
            
            if not isinstance(self.config.emb_dim , int) or self.config.emb_dim <= 0:
                raise ValueError(f"Embedding Dimension must be positive value")
                  
            if not isinstance(self.config.n_heads , int) or self.config.n_heads <= 0:
                raise ValueError(f"Number of heads must be positive value")
            
            if not isinstance(self.config.n_layers , int) or self.config.n_layers <= 0:
                raise ValueError(f"No of Layers must be positive value")
            
            if not isinstance(self.config.drop_rate , float) or not (0.0 <= self.config.drop_rate <= 1.0):
                raise ValueError(f"Drop rate must be in between 0.0 to 1.0")

                        
            return self.config

        except FileNotFoundError:
            raise ValueError(f"Config file not found at {self.Config_path}")
        except JSONDecodeError:
            raise ValueError(f'Not valid json File. Decoding is not posible')