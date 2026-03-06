import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from einops import repeat
from layers.Attn_Bias import BinaryAttentionBias
from layers.Attn_Projection import QueryKeyProjection, RotaryProjection
from utils.masking import TriangularCausalMask, TimerMultivariateMask, TimerCovariateMask



class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100, covariate=False, flash_attention=True):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, attn_mask, n_vars, n_tokens, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars)
        seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)

        queries, keys = self.qk_proj(
            queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(
                        B, n_vars, n_tokens, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(
                        B, n_vars, n_tokens, device=queries.device)
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None

class TemporalAttention(nn.Module):
    def __init__(self,scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100,flash_attention=False):
        super(TemporalAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)

    def forward(self, queries, keys, values, n_vars, n_tokens,attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens)
        seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)

        queries, keys = self.qk_proj(
            queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1).to(queries.device)
        # print("q,k,V:{},{},{}".format(torch.isnan(queries).any(),torch.isnan(keys).any(),torch.isnan(values).any()))

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None

class PatternAndVariateAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, flash_attention=True):
        super(PatternAndVariateAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.flash_attention = flash_attention
    def forward(self, queries, keys, values, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)


        scale = self.scale or 1. / sqrt(E)


        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            # scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None

class HyperEdgeToVertex(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100, flash_attention=True):
        super(HyperEdgeToVertex, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.flash_attention = flash_attention

    def forward(self, queries, keys, values, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        B_v, S, _, D = values.shape

        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        scale = self.scale or 1. / sqrt(E)

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None



class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None,n_vars=None, n_tokens=None, tau=None, delta=None):
        B, L, _ = queries.shape
        Bk, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            n_vars=n_vars,
            n_tokens=n_tokens,
            tau=tau,
            delta=delta,
            attn_mask=attn_mask
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

class HyperGraphAggregateBlock(nn.Module):
    def __init__(self,d_model,d_ff=None,num_heads=8,activation = 'relu',Learned_Q=False,edge_num=4):
        super(HyperGraphAggregateBlock,self).__init__()
        if activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
        if not d_ff:
            d_ff = 4*d_model
        
        self.Learned_Q = Learned_Q
        if Learned_Q:
            self.edge_weight = nn.Parameter(torch.zeros(1,edge_num, d_model))
            nn.init.normal_(self.edge_weight, mean=0.0, std=1.0)
        self.num_heads = num_heads
        # if not Learned_Q:
        self.q_linear = nn.Linear(d_model,d_model)
        self.k_linear = nn.Linear(d_model,d_model)
        self.v_linear = nn.Linear(d_model,d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.FFN = nn.Sequential(
            nn.Linear(d_model,d_ff),
            self.act,
            nn.Linear(d_ff,d_model)
        )

    def forward(self,queries=None,keys=None,values=None):
        if self.Learned_Q:
            queries = self.edge_weight
        H = self.num_heads
        B_q,L,_ = queries.shape
        B_kv,S,_ = values.shape
        # mask generate
        q_norm = queries / (queries.norm(dim=-1, keepdim=True) + 1e-8)  # [B, E, D]
        k_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)        # [B, N, D]
        cos_sim = torch.einsum('bed,bnd->ben', q_norm, k_norm)  # [B, E, N]
        adj_prob = (torch.sigmoid(cos_sim)).detach()  # [B, E, N]
        topk = max(1,adj_prob.shape[1]//3)
        topk_values, topk_indices = torch.topk(adj_prob, topk, dim=1)
        adj = torch.zeros_like(adj_prob)
        adj = adj.scatter_(1, topk_indices, 1)

        queries = self.q_linear(queries)
        keys = self.k_linear(keys)
        values = self.v_linear(values)

        mask = (((1-adj)*-1e9).unsqueeze(1)).detach()

        queries = queries.reshape(B_q,L,H,-1).permute(0,2,1,3) #[B_q,H,L,E]
        keys = keys.reshape(B_kv,S,H,-1).permute(0,2,1,3) #[B_kv,H,S,E]
        values = values.reshape(B_kv,S,H,-1).permute(0,2,1,3) #[B_kv,H,S,E]
        out = torch.nn.functional.scaled_dot_product_attention(
            query = queries,
            key = keys,
            value = values,
            attn_mask = mask,
        )
        out = self.norm1(out.permute(0,2,1,3).reshape(B_kv,L,-1))
        return self.norm2(out+self.FFN(out)),0



