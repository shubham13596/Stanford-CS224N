from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    # Add-norm for multi-head attention.
    if dense_layer == 'attention':
      x = self.attention_dense(input) # 24 x512 xx 768 x 768
      x = dropout(x)
    else:
      x = self.out_dense(input)
      x = dropout(x)
    
    return output + x


  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """
    residual = hidden_states
    hidden_states = self.attention_layer_norm(hidden_states) # b t C  b d h T
    hidden_states = self.self_attention(hidden_states, attention_mask) # b h t d 
    hidden_states = self.add(hidden_states, residual, 'attention', self.attention_dropout)
    
    #store for second residual
    residual = hidden_states
    # Apply layer norm before feed-forward
    hidden_states = self.out_layer_norm(hidden_states)
    hidden_states = self.interm_dense(hidden_states)
    hidden_states = self.interm_af(hidden_states)
    hidden_states = self.add(hidden_states, residual, 'out', self.out_dropout)

    return hidden_states
  
