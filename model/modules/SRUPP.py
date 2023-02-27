import copy
import warnings
import math
from typing import List, Tuple, Union, Optional, Sequence, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from .SRU import SRUCell

#%% SRU++
class SRUppProjectedLinear(nn.Module):
    """
    Projected linear module used in SRU++ module.
    """

    __constants__ = ['in_features', 'out_features', 'proj_features']

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 proj_features: int,
                 dropout: float = 0.0,
                 layer_norm: bool = False):
        """Initialize the projected linear module.

        Parameters
        ----------
        in_features: int
            the number of input features.
        out_features: int
            the number of output features.
        proj_features: int
            the number of features used for attention computation. The input is projected into
            this dimension first. After that the module apply the query-key-value attention
            computation. The output is projected to dimension `out_features`.
        dropout: float, optional
            dropout probability applied after attention computation and before the final projection
            (default=0.0).
        layer_norm: bool, optional
            whether to apply layer normalization within the projected linear module.
        """
        super(SRUppProjectedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj_features = proj_features
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features, proj_features, bias=False)
        self.linear2 = nn.Linear(proj_features, out_features, bias=False)
        self.layer_norm: Optional[nn.Module] = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(proj_features)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_()
        if self.linear2.bias is not None:
            self.linear2.bias.data.zero_()
        if self.dropout.p > 0:
            self.linear2.weight.data.mul_((1 - self.dropout.p)**0.5)

    def forward(self,
                input: Tensor,
                mask_pad: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_mask_pad: Optional[Tensor] = None) -> Tensor:
        """The forward method.
        """
        output = self.linear1(input)
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = self.linear2(self.dropout(output))
        return output


class SRUppAttention(nn.Module):
    """
    Self-attention module used in SRU++ module.
    """

    __constants__ = ['in_features', 'out_features', 'proj_features', 'num_heads',
                     'attn_dropout', 'rezero_init_alpha']

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 proj_features: int,
                 num_heads: int = 1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 rezero_init_alpha: float = 0.0,
                 layer_norm: bool = False,
                 normalize_after: bool = True):
        """Initialize the self-attention module.

        Parameters
        ----------
        in_features: int
            the number of input features.
        out_features: int
            the number of output features.
        proj_features: int
            the number of features used for attention computation. The input is projected into
            this dimension first. After that the module apply the query-key-value attention
            computation. The output is projected to dimension `out_features`.
        num_heads: int, optional
            the number of attention heads used. `proj_features` must be multipler of this value
            (default=1).
        dropout: float, optional
            dropout probability applied after attention computation and before the final projection
            (default=0.0).
        attn_dropout: float, optional
            dropout probability applied on attention map.
        rezero_init_alpha: float, optional
            initial scalar value for the attention transformation `x + alpha * Attention(x)`
            (default=0).
        normalize_after: bool, optional
            if True, apply post layer normalization; otherwise apply pre layer normalization
            (default=True).

        """
        super(SRUppAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj_features = proj_features
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = float(attn_dropout)
        self.rezero_init_alpha = float(rezero_init_alpha)
        self.linear1 = nn.Linear(in_features, proj_features, bias=False)
        self.linear2 = nn.Linear(proj_features, proj_features * 2, bias=False)
        self.linear3 = nn.Linear(proj_features, out_features, bias=False)
        self.alpha = nn.Parameter(torch.Tensor([float(rezero_init_alpha)]))  # type: ignore
        self.normalize_after = normalize_after
        self.layer_norm: Optional[nn.Module] = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(proj_features)

        if proj_features % num_heads != 0:
            raise ValueError("proj_features ({}) must be divisible by num_heads ({})".format(
                proj_features, num_heads
            ))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.alpha.data[:] = self.rezero_init_alpha
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_()
        if self.linear2.bias is not None:
            self.linear2.bias.data.zero_()
        if self.linear3.bias is not None:
            self.linear3.bias.data.zero_()
        if self.dropout.p > 0:
            self.linear3.weight.data.mul_((1 - self.dropout.p)**0.5)

    def forward(self,
                input: Tensor,
                mask_pad: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_mask_pad: Optional[Tensor] = None) -> Tensor:
        """The forward method of SRU++ attention.
        """

        src_len = tgt_len = input.size(0)
        bsz = input.size(1)
        in_dim = input.size(2)
        proj_dim = self.proj_features
        num_heads = self.num_heads
        head_dim = proj_dim // num_heads
        scaling = float(head_dim) ** -0.5

        # concat memory and input as the key-value block when provided
        if memory is not None:
            if memory.dim() != 3 or list(memory.size()[-2:]) != [bsz, in_dim]:
                raise ValueError("memory has size {} but expect {}.".format(
                    list(memory.size()),
                    ['*', bsz, in_dim]
                ))
            mem_len = memory.size(0)
            src_len = memory.size(0) + input.size(0)
            input_ = torch.cat([memory, input], dim=0)
            z = self.linear1(input_)
            residual = z[memory.size(0):]
            layer_norm = self.layer_norm
            if layer_norm is not None:
                if not self.normalize_after:
                    z = layer_norm(z)
            q = z[memory.size(0):]
        else:
            mem_len = 0
            z = residual = self.linear1(input)
            layer_norm = self.layer_norm
            if layer_norm is not None:
                if not self.normalize_after:
                    z = layer_norm(z)
            q = z

        # query, key, value
        k, v = self.linear2(z).chunk(2, dim=-1)
        q = q.contiguous().view(tgt_len, -1, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, -1, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, -1, head_dim).transpose(0, 1)

        # (bsz * num_heads, tgt_len, src_len)
        q = q * scaling
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if list(attn_mask.size()) != [tgt_len, src_len]:
                raise ValueError("attn_mask has size {} but expect {}.".format(
                    list(attn_mask.size()),
                    [tgt_len, src_len]
                ))
            attn_output_weights += attn_mask.unsqueeze(0)

        if mask_pad is not None or memory_mask_pad is not None:
            if mask_pad is None:
                mask_pad = input.new_zeros(tgt_len, bsz, dtype=torch.bool)
            if mem_len > 0:
                if memory_mask_pad is None:
                    memory_mask_pad = input.new_zeros(mem_len, bsz, dtype=torch.bool)
                mask_pad = torch.cat([memory_mask_pad, mask_pad], dim=0)
            if list(mask_pad.size()) != [src_len, bsz]:
                raise ValueError("mask_pad has size {} but expect {}.".format(
                    list(mask_pad.size()),
                    [src_len, bsz]
                ))
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                mask_pad.transpose(0, 1).unsqueeze(1).unsqueeze(2),  # (bsz, 1, 1, src_len)
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.attn_dropout,
                                        training=self.training)

        # (bsz * num_heads, tgt_len, src_len) x (bsz * num_heads, src_len, head_dim)
        #     ---->  (bsz * num_heads, tgt_len, head_dim)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, proj_dim)

        attn_output = attn_output * self.alpha + residual
        layer_norm = self.layer_norm
        if layer_norm is not None:
            if self.normalize_after:
                attn_output = layer_norm(attn_output)

        # (tgt_len, bsz, out_dim)
        attn_output = self.linear3(self.dropout(attn_output))
        return attn_output


class SRUppCell(SRUCell):
    """
    A single layer of SRU++, inherited from SRUCell module
    """

    def forward(self,
                input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_mask_pad: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """The forward method of the SRU++ layer.
        """

        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("Input must be 2 or 3 dimensional")

        batch_size = input.size(-2)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.output_size, dtype=input.dtype,
                             device=input.device)

        # apply layer norm before activation (i.e. before SRU computation)
        residual = input
        layer_norm = self.layer_norm
        if layer_norm is not None:
            if not self.normalize_after:
                input = layer_norm(input)

        # apply dropout for multiplication
        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch_size, input.size(-1)), self.rnn_dropout)
            input = input * mask.expand_as(input)

        # get the scaling constant; scale_x is a scalar
        scale_val: Optional[Tensor] = None
        scale_val = self.scale_x if self.rescale else None

        # get dropout mask
        mask_c: Optional[Tensor] = None
        if self.training and (self.dropout > 0):
            mask_c = self.get_dropout_mask_((batch_size, self.output_size),
                                            self.dropout)

        # compute U
        # U is (length, batch_size, output_size * num_matrices)
        transform_module = self.transform_module
        U = transform_module(input, mask_pad=mask_pad,
                             attn_mask=attn_mask,
                             memory=memory,
                             memory_mask_pad=memory_mask_pad)
        V = self.weight_c

        # apply elementwise recurrence to get hidden states h and c
        h, c = self.apply_recurrence(U, V,
                                     residual, c0,
                                     scale_val,
                                     mask_c,
                                     mask_pad)

        layer_norm = self.layer_norm
        if layer_norm is not None:
            if self.normalize_after:
                h = layer_norm(h)
        return h, c


class SRUpp(nn.Module):
    """
    Implementation of SRU++ module.
    """

    __constants__ = ['input_size', 'hidden_size', 'proj_size', 'output_size',
                     'num_layers', 'num_heads', 'dropout', 'bidirectional',
                     'use_layer_norm', 'num_directions', 'nn_rnn_compatible_return',
                     'input_to_hidden', 'rnn_lst', 'normalization_type']

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 proj_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 num_heads: int = 1,
                 bidirectional: bool = False,
                 layer_norm: bool = False,
                 normalize_after: bool = False,
                 attn_layer_norm: bool = True,
                 highway_bias: float = -2.0,
                 attention_every_n_layers: int = 1,
                 attention_last_n_layers: int = -1,
                 rescale: bool = False,
                 nn_rnn_compatible_return: bool = False,
                 proj_input_to_hidden_first: bool = False,
                 weight_c_init: float = 1.0):
        """Initialize the SRU++ module.

        Parameters
        ----------
        input_size: int
            the number of input features.
        hidden_size: int
            the number of features in the hidden state *for each direction*.
        proj_size: int
            the number of features used for attention.
        num_layers: int, optional
            the number of stacked SRU++ layers (default=2).
        dropout: float, optional
            dropout probability applied between sub-layers (default=0.0).
        attn_dropout: float, optional
            dropout probability applied on attention map (default=0.0).
        num_heads: int, optional
            number of attention heads (default=1).
        bidirectional: bool, optional
            if True, use bidirectional SRU++ (default=False).
        layer_norm: bool, optional
            whether to apply layer normalization to each SRU++ layer (default=False).
        normalize_after: bool, optional
            whether to apply post layer norm that normalizes the output of each SRU++ layer
            (default=False).
        attn_layer_norm: bool, optional
            whether to apply layer norm in the attention module or projected linear module if
            attention is disabled (default=True).
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid) gate (default=-1.0).
        attention_every_n_layers: int, optional
            only introduce attention every few layers of SRU++. by default, every SRU++ layer has
            attention (default=1).
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the residual term (default=False).
        proj_input_to_hidden_first: bool, optional
            if True, apply an nn.Linear module to the input of this module when input_size !=
            hidden_size (default=False).
        weight_c_init: float, optional
            size of uniform initiatialization of weight_c
            (default=1.0)

        """
        if attention_every_n_layers != 1 and attention_last_n_layers != -1:
            raise ValueError("Cannot set both attention_every_n_layers and "
                             "attention_last_n_layers in SRU++ module.")
        super(SRUpp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.num_directions = 2 if bidirectional else 1
        self.nn_rnn_compatible_return = nn_rnn_compatible_return
        self.input_to_hidden: Optional[nn.Module] = None
        if proj_input_to_hidden_first and input_size != self.output_size:
            first_layer_input_size = self.output_size
            self.input_to_hidden = nn.Linear(input_size, self.output_size, bias=False)
            nn.init.xavier_uniform_(self.input_to_hidden.weight)
        else:
            first_layer_input_size = input_size

        # attention configuration
        if attention_last_n_layers != -1:
            use_attention = lambda ind: num_layers - ind <= attention_last_n_layers  # noqa
        else:
            use_attention = lambda ind: (ind + 1) % attention_every_n_layers == 0  # noqa

        for i in range(num_layers):
            # create the i-th SRU layer
            in_features = first_layer_input_size if i == 0 else self.output_size
            proj_features = proj_size
            out_features = self.output_size * (3 if in_features == self.output_size else 4)
            custom_m: Optional[nn.Module] = None
            if use_attention(i):
                custom_m = SRUppAttention(
                    in_features,
                    out_features,
                    proj_features,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    num_heads=num_heads,
                    layer_norm=attn_layer_norm,
                )
            else:
                custom_m = SRUppProjectedLinear(
                    in_features,
                    out_features,
                    proj_features,
                    dropout=dropout,
                    layer_norm=attn_layer_norm,
                )
            layer = SRUppCell(
                in_features,
                self.hidden_size,
                dropout=dropout if i + 1 != num_layers else 0,
                bidirectional=bidirectional,
                layer_norm=layer_norm,
                normalize_after=normalize_after,
                highway_bias=highway_bias,
                rescale=rescale,
                transform_module=custom_m,
                weight_c_init=weight_c_init,
            )
            self.rnn_lst.append(layer)

    def __getitem__(self, n: int) -> SRUppCell:
        """
        returns n'th layer SRUppCell
        """
        return self.rnn_lst[n]

    def forward(self,
                input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                memory: Optional[List[Optional[Tensor]]] = None,
                memory_mask_pad: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Tensor, Dict[str, List[Tensor]]]:
        """
        The forward method of SRUpp module.

        Parameters
        ----------
        input: Tensor
            the input feature. shape: (length, batch_size, input_size)
        c0: Tensor, optional
            the initial internal hidden state. shape: (num_layers,
            batch_size, output_size) where
            output_size = hidden_size * num_direction
        mask_pad: Tensor, optional
            the mask where a non-zero value indicates if an input token
            is pad token that should be ignored in forward and backward
            computation. shape: (length, batch_size)
        attn_mask: Tensor, optional
            the additive attention mask. shape: (input_length, context_length)
            the mask is a float tensor that will be directly added to the
            attention weights before softmax normalization.
            input_length is the length of the input tensor, and context length
            is the total length of context states that each input can attend
            to. the context_length is equal to the sum of input_length and the
            lengths of extra memory states given by `memory`.
        memory: a list of optional tensors, optional
            a list of memory tensors as additional inputs for the attention
            to attend to. the size of the list is equal to the number of layers
            of SRUpp module. memory[i] is the memory tensor for the (i+1)-th
            layer and its second dimension (batch size) and third dimension
            (hidden size) must be compatible with the input tensor to the
            (i+1)-th layer.
        memory_mask_pad: tensor, optional
            the mask tensor indicate if a position in the memory tensors is
            an invalid / pad token that should be ignored in attention.
            shape: (memory_length, batch_size)

        Returns
        ----------
        h: Tensor
            the output hidden state. shape: (length, batch_size,
            output_size) where
            output_size = hidden_size * num_direction
        c: Tensor
            the last internal hidden state. shape: (num_layers,
            batch_size, output_size), or (num_layers * num_directions,
            batch_size, hidden_size) if `nn_rnn_compatible_return` is
            set `True`
        memory_bank: Dict[str, List[Tensor]]
            a dictionary that stores various internal states indexed
            by state names. each value is a list of tensors in which
            the i-th element is the state tensor of the (i+1)-th layer.
            these internal states can be reused for attention for the
            next forward call during training and decoding.

        """
        # unpack packed, if input is packed. packing and then unpacking will be slower than not
        # packing at all, but makes SRU++ usage compatible with nn.RNN usage
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, lengths = nn.utils.rnn.pad_packed_sequence(input)
            max_length = lengths.max().item()
            mask_pad = torch.ByteTensor([[0] * length + [1] * (max_length - length)
                                        for length in lengths.tolist()])
            mask_pad = mask_pad.to(input.device).transpose(0, 1).contiguous()

        # The dimensions of `input` should be: `(sequence_length, batch_size, input_size)`.
        if input.dim() != 3:
            raise ValueError("There must be 3 dimensions for (length, batch_size, input_size)")

        length = input.size(0)
        bsz = input.size(1)
        input_size = input.size(2)
        num_layers = self.num_layers
        output_size = self.output_size

        if input_size != self.input_size:
            raise ValueError(f"Input has size (*, *, {input_size}) but expect a last dimension of {self.input_size}")

        if c0 is None:
            zeros = torch.zeros(bsz, output_size, dtype=input.dtype, device=input.device)
            c0_ = [zeros for i in range(num_layers)]
        else:
            if list(c0.size()) != [num_layers, bsz, output_size]:
                raise ValueError(f"c0 has size {list(c0.size())} but expect {[num_layers, bsz, output_size]}.")
            c0_ = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]

        if mask_pad is not None and list(mask_pad.size()) != [length, bsz]:
            raise ValueError(f"mask_pad has size {list(mask_pad.size())} but expect {[length, bsz]}.")

        if memory is not None and not isinstance(memory, list):
            raise ValueError(f"memory has type {type(memory)} but expect List[Tensor].")

        if memory is not None and len(memory) != num_layers:
            raise ValueError(f"memory has size {len(memory)} but expect {num_layers}.")

        x = input if self.input_to_hidden is None else self.input_to_hidden(input)
        prev_inputs, lstc = [], []
        i = 0
        x = x.contiguous()

        for rnn in self.rnn_lst:
            prev_inputs.append(x)
            memory_i = memory[i] if memory is not None else None
            h, c = rnn(x, c0_[i],
                       mask_pad=mask_pad,
                       attn_mask=attn_mask,
                       memory=memory_i,
                       memory_mask_pad=memory_mask_pad)
            x = h
            lstc.append(c)
            i += 1

        lstc_stack = torch.stack(lstc)
        if self.nn_rnn_compatible_return:
            lstc_stack = lstc_stack.view(num_layers, bsz, self.num_directions, self.hidden_size)
            lstc_stack = lstc_stack.transpose(1, 2).contiguous()
            lstc_stack = lstc_stack.view(num_layers * self.num_directions, bsz, self.hidden_size)

        if isinstance(orig_input, PackedSequence):
            h = nn.utils.rnn.pack_padded_sequence(h, lengths, enforce_sorted=False)

        return (h, lstc_stack, {'saved_inputs': prev_inputs})

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()
        if self.input_to_hidden is not None:
            nn.init.xavier_uniform_(self.input_to_hidden.weight)
