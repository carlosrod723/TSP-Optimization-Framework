# models/pointer_network/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class PointerAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # decoder_state: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        encoder_transform = self.W1(encoder_outputs)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)
        scores = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Return attention weights
        return F.softmax(scores, dim=-1)

class PointerNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder_rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Decoder
        self.decoder_rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size * 2,  # Bidirectional encoder
            num_layers=num_layers,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = PointerAttention(hidden_size * 2)
        
    def forward(
        self,
        inputs: torch.Tensor,
        start_node: Optional[int] = None,
        teacher_forcing: bool = False,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # Encode inputs
        encoder_outputs, (hidden, cell) = self.encoder_rnn(inputs)
        
        # Initialize decoder input (start with zeros or specified node)
        if start_node is not None:
            decoder_input = inputs[:, start_node:start_node+1]
        else:
            decoder_input = torch.zeros_like(inputs[:, 0:1])
        
        # Initialize outputs
        outputs = []
        mask = torch.ones(batch_size, seq_len, device=inputs.device)
        
        # Decoding steps
        for i in range(seq_len):
            # Run decoder for one step
            decoder_output, (hidden, cell) = self.decoder_rnn(
                decoder_input,
                (hidden, cell)
            )
            
            # Calculate attention
            attn_weights = self.attention(
                decoder_output.squeeze(1),
                encoder_outputs,
                mask
            )
            outputs.append(attn_weights)
            
            # Update mask to prevent revisiting nodes
            if i < seq_len - 1:
                selected_node = attn_weights.max(1)[1]
                mask.scatter_(1, selected_node.unsqueeze(-1), 0)
            
            # Prepare next input
            if teacher_forcing and targets is not None:
                decoder_input = inputs.gather(
                    1,
                    targets[:, i:i+1].unsqueeze(-1).expand(-1, -1, inputs.size(-1))
                )
            else:
                decoder_input = inputs.gather(
                    1,
                    attn_weights.max(1)[1].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, inputs.size(-1))
                )
        
        return torch.stack(outputs, dim=1)