'''
Custom LoRA implementation for the CS224N GPT-2 attention module.
This directly handles the einops-based attention implementation.
'''

import torch
import torch.nn as nn
import math
from einops import rearrange


class LoRALinear(nn.Module):
    """
    LoRA implementation for linear layers that keeps the original layer frozen
    and adds trainable low-rank matrices.
    """
    def __init__(self, original_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        
        # Dimensions
        self.in_features = original_layer.weight.shape[1]
        self.out_features = original_layer.weight.shape[0]
        self.r = r
        self.lora_alpha = lora_alpha
        
        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros((r, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))
        
        # Optional dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Scaling factor
        self.scaling = lora_alpha / r
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA path
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        # Combine the outputs
        return original_output + lora_output


class LoRACausalSelfAttention(nn.Module):
    """
    Applies LoRA to the CausalSelfAttention module of the CS224N GPT-2 implementation.
    Specifically targets the query, key, and value projections.
    """
    def __init__(self, config, original_attn=None):
        super().__init__()
        
        # If we're starting from scratch
        if original_attn is None:
            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            
            # Create regular linear layers
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
            
            # This is the original dropout from the implementation
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        else:
            # If we're wrapping an existing attention layer
            self.num_attention_heads = original_attn.num_attention_heads
            self.attention_head_size = original_attn.attention_head_size
            self.all_head_size = original_attn.all_head_size
            
            # Create LoRA-wrapped projections
            self.query = LoRALinear(
                original_attn.query, 
                r=config.lora_r, 
                lora_alpha=config.lora_alpha, 
                lora_dropout=config.lora_dropout
            )
            self.key = LoRALinear(
                original_attn.key, 
                r=config.lora_r, 
                lora_alpha=config.lora_alpha, 
                lora_dropout=config.lora_dropout
            )
            self.value = LoRALinear(
                original_attn.value, 
                r=config.lora_r, 
                lora_alpha=config.lora_alpha, 
                lora_dropout=config.lora_dropout
            )
            
            # Use the original dropout
            self.dropout = original_attn.dropout
            
    def transform(self, x, linear_layer):
        """
        Transform function that projects and reshapes for multi-head attention.
        Identical to the original implementation but uses our LoRA layers.
        """
        # Apply projection
        proj = linear_layer(x)
        
        # Reshape as in the original implementation using einops
        proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
        proj = rearrange(proj, 'b t h d -> b h t d')
        
        return proj
        
    def attention(self, key, query, value, attention_mask):
        """
        Attention calculation, keeping the same implementation as the original.
        """
        # Get attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.attention_head_size ** 0.5)
        
        # Apply causal mask
        mask = torch.tril(torch.ones_like(attn_scores))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply any additional mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        # Get attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to value
        y = torch.matmul(attn_probs, value)
        y = rearrange(y, 'b h t d -> b t (h d)')
        
        return y
        
    def forward(self, hidden_states, attention_mask):
        """
        Forward pass identical to the original but using our LoRA components.
        """
        # Generate projections with LoRA
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        
        # Calculate attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        
        return attn_value


def apply_lora_to_gpt2_layer(layer, config):
    """
    Apply LoRA to a GPT2Layer by replacing the attention components.
    
    Args:
        layer: The GPT2Layer to modify
        config: Configuration with LoRA parameters
        
    Returns:
        The modified layer with LoRA applied
    """
    # Create a LoRA version of the self-attention
    lora_self_attention = LoRACausalSelfAttention(config, original_attn=layer.self_attention)
    
    # Replace the original attention
    layer.self_attention = lora_self_attention
    
    # Also create a LoRA version of the attention dense layer
    layer.attention_dense = LoRALinear(
        layer.attention_dense,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )
    
    # The rest of the layer remains unchanged and frozen
    for param in layer.interm_dense.parameters():
        param.requires_grad = False
    for param in layer.out_dense.parameters():
        param.requires_grad = False
    for param in layer.attention_layer_norm.parameters():
        param.requires_grad = False
    for param in layer.out_layer_norm.parameters():
        param.requires_grad = False
    
    return layer


def apply_lora_to_gpt2(model, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Apply LoRA to a GPT-2 model by modifying its layers.
    
    Args:
        model: The GPT-2 model to modify
        lora_r: Rank of the LoRA decomposition
        lora_alpha: Scaling factor
        lora_dropout: Dropout probability
        
    Returns:
        The modified model with LoRA applied
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA config to the model config
    model.config.lora_r = lora_r
    model.config.lora_alpha = lora_alpha
    model.config.lora_dropout = lora_dropout
    
    # Apply LoRA to each layer
    for i in range(len(model.gpt_layers)):
        model.gpt_layers[i] = apply_lora_to_gpt2_layer(model.gpt_layers[i], model.config)
    
    # Count and print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Applied LoRA to GPT-2 model")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    return model