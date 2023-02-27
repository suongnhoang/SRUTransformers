import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
        super().__init__()
        self.W = nn.Linear(in_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        nn.init.xavier_normal_(self.W.weight)
        nn.init.constant_(self.W.bias, 0)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.constant_(self.V.bias, 0)
        
    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att) # [batch, seq_len, 1]
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1) # [batch, dim]
        return context_vector
