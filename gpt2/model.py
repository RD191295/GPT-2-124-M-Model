import torch
import torch.nn as nn
from gpt2.layers import FeedForward,LayerNorm,MultiHeadAttention

class TransformerBlock(nn.Module):
    """
    Transformer Block for GPT-2

    Implements a single Transformer block as used in GPT-2, consisting of:
        - Layer normalization before self-attention (pre-norm architecture)
        - Multi-head self-attention with causal masking
        - Residual connection around the attention layer
        - Layer normalization before the feed-forward network
        - Position-wise feed-forward network (MLP with GELU activation)
        - Residual connection around the feed-forward layer

    Parameters
    ----------
    d_in : int
        Input dimensionality (embedding size).
    d_out : int
        Output dimensionality (embedding size, same as d_in for GPT-2).
    n_heads : int
        Number of attention heads in the multi-head attention mechanism.
    context_length : int
        Maximum sequence length (used for causal masking in attention).
    dropout : float
        Dropout probability applied after attention and feed-forward layers.
    qkv_bias : bool, optional (default=False)
        Whether to include bias terms in the query, key, and value projections.

    Inputs
    ------
    x : torch.Tensor of shape (batch_size, sequence_length, d_in)
        Input hidden states to the Transformer block.
    
    Returns
    -------
    torch.Tensor of shape (batch_size, sequence_length, d_out)
        Output hidden states after attention and feed-forward transformations.

    Notes
    -----
    - Uses "pre-norm" Transformer architecture (LayerNorm before each sub-layer).
    - Causal masking ensures that each position attends only to past and current tokens.
    """
    def __init__(self,d_in:int, d_out:int,n_heads:int,
                 context_length:int,dropout:float,qkv_bias:bool=False) -> None:
        super().__init__() # type: ignore
        assert (d_in == d_out),\
                "d_in and d_out should be same"
        
        self.attention = MultiHeadAttention(d_in=d_in,
                                            d_out=d_out,
                                            context_length=context_length,
                                            num_of_heads=n_heads,
                                            dropout=dropout,
                                            qkv_bias=qkv_bias)
        self.norm_layer1 = LayerNorm(d_in)
        self.norm_layer2 = LayerNorm(d_in)

        self.feedforward = FeedForward(d_in)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self , x :torch.Tensor)->torch.Tensor:
        """
        Forward pass of the Transformer block.

        Applies layer normalization, multi-head self-attention with causal masking,
        residual connections, feed-forward network, and dropout.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, sequence_length, d_in)
            Input hidden states from embeddings or previous transformer block.
        
        Returns
        -------
        torch.Tensor of shape (batch_size, sequence_length, d_in)
            Transformed hidden states after applying attention and feed-forward layers.
        """
        shortcut = x
        x = self.norm_layer1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm_layer2(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPT2Model(nn.Module):
    """
    GPT-2 Language Model

    Implements the core GPT-2 Transformer architecture as described in 
    "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019).

    The model consists of:
        - Token embeddings
        - Positional embeddings
        - A stack of Transformer blocks (multi-head self-attention + feed-forward layers)
        - Final layer normalization
        - Linear projection head to vocabulary logits

    Parameters
    ----------
    vocab_size : int
        Size of the tokenizer vocabulary (default: 50257 for GPT-2).
    context_length : int
        Maximum sequence length (default: 1024).
    emb_dim : int
        Dimensionality of token embeddings and hidden states.
    n_heads : int
        Number of attention heads per Transformer block.
    n_layers : int
        Number of Transformer blocks.
    dropout : float
        Dropout probability applied to embeddings, attention, and feed-forward layers.
    qkv_bias : bool, optional (default=False)
        Whether to include bias terms in the query, key, and value projections.

    Inputs
    ------
    input_ids : torch.LongTensor of shape (batch_size, sequence_length)
        Indices of input tokens in the vocabulary.

    Returns
    -------
    logits : torch.Tensor of shape (batch_size, sequence_length, vocab_size)
        Prediction scores (unnormalized logits) for each token over the vocabulary.

    Notes
    -----
    - The model applies causal (left-to-right) masking so that each position 
      can only attend to tokens at or before its position.
    """
    def __init__(self,vocab_size:int,context_length:int,
                 emb_dim:int,n_heads:int,
                 n_layers:int,dropout:float,
                 qkv_bias:bool=False) -> None:
        super().__init__() # type: ignore
        self.tok_emb = nn.Embedding(vocab_size,emb_dim)
        self.pos_emb = nn.Embedding(context_length,emb_dim)
        self.drop_emb = nn.Dropout(dropout)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(emb_dim,emb_dim,n_heads,context_length,dropout,qkv_bias) 
              for _ in range(n_layers)])
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(
            emb_dim, vocab_size , bias=qkv_bias
        )
        
    def forward(self, input_ids :torch.Tensor)->torch.Tensor:
        """
        Forward pass of the GPT-2 model.

        Converts input token indices into embeddings, adds positional encodings,
        applies a stack of Transformer blocks, and returns the logits.

        Parameters
        ----------
        input_ids : torch.LongTensor of shape (batch_size, sequence_length)
            Indices of input tokens in the vocabulary.

        Returns
        -------
        logits : torch.Tensor of shape (batch_size, sequence_length, vocab_size)
        Prediction scores (logits) for each token over the vocabulary.

        """
        _, seq_len = input_ids.shape # type: ignore
        tok_embs = self.tok_emb(input_ids)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=input_ids.device)) 
        in_embs = tok_embs + pos_embs
        x = self.drop_emb(in_embs)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


