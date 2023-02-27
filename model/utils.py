import torch
from torch import nn
import torch.nn.functional as F 

class WeightsInitializer:
    def __init__(self, config = None):
        self.config = config

    def set_config(self, config):
        self.config = config

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_attention_function(q, k, v, activation="softmax"):
    """
     q: [bs, length, dim] or [bs, , dim]
     k=v: [bs, length, dim] or [bs, ans_seq, dim]
    """
    assert activation.lower() in ["softmax", "tanh", "sigmoid"],\
        'activation must one of ["softmax", "tanh", "sigmoid"]'
    
    if len(q.shape) == 2:
        q = q.unsqueeze(dim=0)
    if len(k.shape) == 2:
        k = k.unsqueeze(dim=0)
    if len(v.shape) == 2:
        v = v.unsqueeze(dim=0)
    
    attn_logits = torch.matmul(q, k.transpose(2, 1))/k.shape[-1] # [bs, length, length] or [bs, query_seq, ans_seq]
    
    if activation == "softmax":
        attn_weights = torch.softmax(attn_logits, -1)
    elif activation == "tanh":
        attn_weights = torch.tanh(attn_logits)
    elif activation == "sigmoid":
         attn_weights = torch.sigmoid(attn_logits)
    
    output = torch.matmul(attn_weights, v) # [bs, length, dim] or [bs, query_seq, dim]
    return output