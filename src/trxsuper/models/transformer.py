# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications
"""
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .swin_unetr import SwinUNETR
from .position_encoding import PositionEmbeddingSine3D


class SwinUneTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False,
                 img_size=[32, 32, 32], in_channels=2, out_channels=96,
                 depths=[2, 2, 2, 2], num_heads=[2, 4, 8, 16], feature_size=24,
                 patch_size=1, window_size=8, mem_vol_ds=1, focus_vol_size=False,
                 take_in_between_level=False):
        super().__init__()

        assert focus_vol_size < img_size[0], "focus_vol_size should be less than sub_vol_size!"
        swin_unetr_kwargs = {
            'img_size': img_size,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'depths': depths,
            'num_heads': num_heads,
            'feature_size': feature_size,
            'patch_size': patch_size,
            'window_size': window_size,
            'mem_vol_ds': mem_vol_ds,
            'take_in_between_level': take_in_between_level
        }
        _encoder = SwinUNETR(**swin_unetr_kwargs)
        self.encoder = _encoder
        self.pos_encoder = PositionEmbeddingSine3D(d_model, normalize=True)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.image_size = img_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.focus_vol_size = focus_vol_size

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, query_embed, pos, src=None, memory=None, tgt_mask=None):
        if memory is None:
            memory = self.encoder(src)
            if self.focus_vol_size:
                img_size = torch.tensor(memory.size()[-3:])
                s_idx = (img_size - self.focus_vol_size) // 2
                e_idx = s_idx + self.focus_vol_size
                memory = memory[:, :, s_idx[0]:e_idx[0], s_idx[1]:e_idx[1], s_idx[2]:e_idx[2]]
            pos = self.pos_encoder(memory)
            memory = memory.flatten(2).permute(2, 0, 1)
            pos = pos.flatten(2).permute(2, 0, 1)

        hs, hs_without_norm = self.decoder(tgt, memory, query_pos=query_embed, pos=pos, tgt_key_padding_mask=tgt_mask)
        return hs.transpose(1, 2), hs_without_norm.transpose(1, 2), memory, pos


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        intermediate_self_weights = []
        intermediate_cross_weights = []

        for layer in self.layers:
            if layer.need_weights:
                output, self_weights, cross_weights = layer(output, memory, tgt_mask=tgt_mask,
                                                            memory_mask=memory_mask,
                                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                                            memory_key_padding_mask=memory_key_padding_mask,
                                                            pos=pos, query_pos=query_pos)
                if self.return_intermediate:
                    intermediate_self_weights.append(self_weights)
                    intermediate_cross_weights.append(cross_weights)
            else:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            output = torch.stack(intermediate)

        if self.norm is not None:
            norm_output = self.norm(output)
        else:
            norm_output = output

        if len(intermediate_self_weights) > 0:
            return norm_output, output, torch.stack(intermediate_self_weights), torch.stack(intermediate_cross_weights)
        else:
            return norm_output, output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, need_weights=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.need_weights = need_weights

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.need_weights:
            tgt2, self_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask, need_weights=self.need_weights)
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask, need_weights=self.need_weights)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.need_weights:
            tgt2, cross_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                                      key=self.with_pos_embed(memory, pos),
                                                      value=memory, attn_mask=memory_mask,
                                                      key_padding_mask=memory_key_padding_mask, need_weights=self.need_weights)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask, need_weights=self.need_weights)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if self.need_weights:
            return tgt, self_weights, cross_weights
        else:
            return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask, need_weights=self.need_weights)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, need_weights=self.need_weights)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return SwinUneTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        img_size=[args.sub_vol_size] * 3,
        in_channels=1 if args.past_tr_type == 'prev_pos' else 2,
        out_channels=args.hidden_dim,
        depths=args.depths,
        num_heads=args.num_heads,
        feature_size=args.unetr_dim,
        patch_size=args.patch_size,
        window_size=args.window_size,
        mem_vol_ds=args.mem_vol_ds,
        focus_vol_size=args.focus_vol_size,
        take_in_between_level=args.take_in_between_level
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
