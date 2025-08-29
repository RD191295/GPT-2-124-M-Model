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
            
            if not hasattr(self.config, 'vocab_size'):
                print("vocab size is not present.")
            if not hasattr(self.config, 'context_length'):
                print("context length is not present.")
            if not hasattr(self.config, 'emb_dim'):
                print("Embedding Dimesion is not present.")
            if not hasattr(self.config, 'n_heads'):
                print("No of Heads is not present.")
            if not hasattr(self.config, 'n_layers'):
                print("No of Layers is not present.")
            if not hasattr(self.config, 'drop_rate'):
                print("Drop Rate is not present.")
            

            

            return self.config

        except FileNotFoundError:
            print(f"Config file not found at {self.Config_path}")
        except JSONDecodeError:
            print(f'Not valid json File. Decoding is not posible')



