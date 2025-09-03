import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Mechanism 
            Multi Head Attention computed as follows:

                MultiHead(Q,K,V)=Concat(head1​,head2​,…,headh​)WO​
                where headi is calulated as follows:

                    headi​=Attention(QWiQ​,KWiK​,VWiV​)

        Multi Head Attention Calculated as follows:
            
            Calculate Query Key value matrix as follows:
            >> Query = Wq @ X
            >> Key =   Wk @ X
            >> Value = Wv @ X

            reshape Query Key and Value:
            >> (batch, num_of_tokens, d_out) --> (batch, num_of_tokens, num_of_heads, head_dim)

            Grouping by heads:
            >> (batch, num_of_tokens, num_of_heads, head_dim) --> (batch, num_of_heads, num_of_tokens,head_dim)

            Calucalting Attention score:
            >> attn_scores = Query @ Key

            Apply Upper Tringle Mask ( Masking Next All token and keeping only previous token):
            >> UpperTriMask(attn_Scores)

            Calculating Attention weight by Softmax and deviding by sqrt(head_dim) this ensure variance remain same
            >> softmax( 1/sqrt(head_dim) * attn_scores)

            Calculating Context vector
            >>  context_vect = (attn_weights @ Value)

            reshape to merge heads
            (batch, num_of_heads, num_of_tokens,head_dim) --->(batch,num_of_tokens,num_of_heads,head_dim)

            Ouput project layer
            >> LinearLayer(batch,num_of_tokens,self.d_out)
            >> (batch,num_of_tokens,num_of_heads,head_dim) ---> batch,num_of_tokens,self.d_out)

            Dropout Layer
            >> context_vector = ropout(context_vector)
            
        attributes:
            d_in(int): Input feature dimension.
            d_out(int): Output feature dimension
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
            torch.triu(torch.ones(context_length, context_length),
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
    

class GELU(nn.Module):
    """GELU Activation Layer 
        GELUs full form is GAUSSIAN ERROR LINEAR UNIT

        GELU Activation is defined as Follows:
                GELU(x) = xP(X ≤ x) = xΦ(x).

        GELU Can be aproximate with following formula,
                GELU(x)  = 0.5x(1 + tanh[sqrt(2/π)(x + 0.044715x3)])

        RELU is common and popular activation function but it has some limitation
        as follows:
            Relu DO not give negative input. so all negative neruon become zero
            that may be loss of some information effectivel lead to dying neruon problem
            Relu can not be differatiable at x = 0 which affect gradient optimization

        GELU is a smoother, non-linear activation function that incorporates a probabilistic
        element based on the Gaussian distribution, allowing for small negative values to pass
        through. Unlike ReLU, which hard-clips negatives, GELU smoothly gates inputs in proportion 
        to their value, making it particularly effective in deep Transformer architectures.

        GELU Avoid above problem and can be differentiable which allow to optimize gredient
        effieciently. 

        GELU is computationally extensive compare to RELU.
        
        methods:
            forward(self,x:torch.Tensor)->int:
                - Forward method for activation function
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            'constant',
            torch.sqrt(torch.tensor(2.0/torch.pi))
        )
    

    def forward(self , x:torch.Tensor)->torch.Tensor:
        """ Forward method for GELU Activation function

            Args:
                x(torch.Tensor): Input Tensor

            
            returns:
                torch.Tensor : Activation function output
        
        """
        return 0.5 * x *(1 + torch.tanh(
            self.constant *(x+ 0.044715* torch.pow(x,3))
            )) 
    


class FeedForward(nn.Module):
    """Feed Forward Neural Network

        Feed Forward Neural Network have two fully connected layer with GELU 
        as activation function.

        Feed Forward network can be described as ,
        
        FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
        
        here w1, w2 are weight matrix and b1 and b2 are bias matrix
        
        Feed forward first layer project linear projection with input x as dim equal to emb_dim
        to output dim as dff that is 4 * emb_dim . this layer work as expansion which help network
        to find useful information as higher dimension allow to adapt more complex features

        Follow by it has non linear activation function GELU allowing to add some non linearity in network.

        Follow by it has contraction layer which have input dimension as dff= 4*emb_dim  and output which dimension
        is emb_dim . at the end it preserve actual dimension of input 

        attributes:
            emb_dim (int) : Embedding Dimension

        methods:
            forward(self , x:torch.Tensor)->torch.Tensor:
                -Forward method for Feed forward Neural network
    """

    def __init__(self, emb_dim:int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            nn.Linear(emb_dim , 4 * emb_dim),
            GELU(),
            nn.Linear(4*emb_dim, emb_dim)
        )
    

    def forward(self ,x:torch.Tensor)->torch.Tensor:
        """Forward Method for Feed forward network

            Args:
                x(torch.Tensor): Input Tensor

            
            returns:
                torch.Tensor : Feed Forward network output module
        
        """
        return self.layers(x)


class LayerNorm(nn.Module):
    """Layer Normalization Layer

        layer normalization ensures “all neurons in a particular layer 
        effectively have the same distribution across all features for a given input.”

        Layer normalization ensure mean = 0 and std deviation 1

        eps (small factor) is added to ensure we do not get division error

        LayerNorm(x) = self.scale *((x-mean)/ (torch.sqrt(var+ self.eps))) + self.shift
        
        LayerNorm uses learnable scale (gamma, γ) and shift (beta, β) parameters to allow 
        the network to adapt the normalized activations, preserving the model's expressive 
        power by enabling it to scale and shift the normalized distribution back if a strict 
        zero-mean, unit-variance constraint is suboptimal for learning. This affine transformation
        allows the network to control the output distribution and ensures that the information within
        the normalized features isn't lost, leading to more stable training and better model performance.

        attributes:
            emb_dim (int) : Embedding Dimension

        methods:
            forward(self , x:torch.Tensor)->torch.Tensor:
                -Layer Normalization forward method
    """

    def __init__(self,emb_dim:int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        mean = x.mean(dim = -1,keepdim=True)
        var = x.var(dim = -1, keepdim=True, unbiased=False)

        norm_x = (x-mean)/ (torch.sqrt(var+ self.eps))

        return self.scale * norm_x +self.shift