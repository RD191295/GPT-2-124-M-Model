import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Mechanism 
            Multi Head Attention computed as follows:

                MultiHead(Q,K,V)=Concat(head1​,head2​,…,headh​)WO​
                where headi is calulated as follows:

                    headi​=Attention(QWiQ​,KWiK​,VWiV​)

        attributes:
            d_in(int): Input feature dimension.
            d_out(int): Output feature dimesion
            num_of_heads(int): Number of attention heads
            context_length(int):Minimul sequence length for causal masking
            dropout(float):Dropout probability applied to output (and optionally attention weights).
            qkv_bias(bool): Weather to apply bias to Q/K/V projection
        
        methods:
            forward(self,x:torch.Tensor)->torch.Tensor:
                - Forward method for calculating context vector(Multi Head attention)
    """
    def __init__(self,d_in:int,
                 d_out:int,
                 context_length:int,
                 num_of_heads:int,
                 dropout:float,
                 qkv_bias:bool=False)->None:
        super().__init__()
        assert (d_out % num_of_heads == 0) , \
                "d_out is not divisible by num of heads"
        
        self.d_out = d_out
        self.num_of_heads = num_of_heads
        self.head_dim = d_out // num_of_heads

        self.W_Query = nn.Linear(d_in, d_out , bias = qkv_bias)
        self.W_Key = nn.Linear(d_in, d_out , bias = qkv_bias)
        self.W_Value = nn.Linear(d_in, d_out , bias = qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length)
                       diagonal = 1)
        )
    

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """Forward Pass for Multi Head attention

            Args:
                x(torch.Tensor): Input Tensor

            
            returns:
                torch.Tensor : Multi Head attention context vector
        
        """
        batch, num_of_tokens, d_in = x.shape
        
        Query = self.W_Query(x)  # (batch, num_of_tokens, d_out)
        Key = self.W_Key(x) # (batch, num_of_tokens, d_out)
        Value = self.W_Value(x) #(batch, num_of_tokens, d_out)

        # reshape to add num_of_heads(batch, num_of_tokens, d_out) -> (batch, num_of_tokens, num_of_heads, head_dim)
        Query = Query.view(batch, num_of_tokens, self.num_of_heads, self.head_dim)
        Key = Key.view(batch, num_of_tokens, self.num_of_heads, self.head_dim)
        Value = Value.view(batch, num_of_tokens, self.num_of_heads, self.head_dim)

        # Grouping by heads (batch, num_of_tokens, num_of_heads, head_dim) -> (batch, num_of_heads,num_of_tokens, head_dim)
        #e.g. (2,6,2,3) (2,6,2,3)
        Query = Query.transpose(1,2)
        Key = Key.transpose(1,2)
        Value = Value.transpose(1,2)

        #calculate attention score
        attn_scores = Query @ Key.transpose(2,3)

        mask_bool = self.mask.bool()[:num_of_tokens,:num_of_tokens]

        attn_scores = attn_scores.masked_fill(mask_bool, float("-inf"))

        # apply softmax
        attn_weights = torch.softmax(
            attn_scores / self.head_dim ** 0.5,
            dim = -1
        )

        # calculate context vector
        context_vect = (attn_weights @ Value).transpose(1,2)

        # merge heads (batch, num_of_tokens, num_of_heads, head_dim) -> (batch, num_of_tokens, d_out)
        context_vect = context_vect.contiguous().view(batch,num_of_tokens,self.d_out)
        context_vect = self.out_proj(context_vect)

        return self.dropout(context_vect)