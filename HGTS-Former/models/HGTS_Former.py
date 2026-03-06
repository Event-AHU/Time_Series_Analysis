import torch
from torch import nn
from layers.Transformer_EncDec import HyperGraphTimeLayer
from layers.SelfAttention_Family import AttentionLayer, TemporalAttention, PatternAndVariateAttention, HyperEdgeToVertex, HyperGraphAggregateBlock
from layers.MLP import HyperEdgeMLP


class Model(nn.Module):
    """
    Timer-XL: Long-Context Transformers for Unified Time Series Forecasting 

    Paper: https://arxiv.org/abs/2410.04803
    
    GitHub: https://github.com/thuml/Timer-XL
    
    Citation: @article{liu2024timer,
        title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
        author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
        journal={arXiv preprint arXiv:2410.04803},
        year={2024}
    }
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.input_token_len = configs.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        self.output_attention = configs.output_attention
        self.blocks = [HyperGraphTimeLayer(
            TemporalAttention=AttentionLayer(
                attention=TemporalAttention(
                    d_model=configs.d_model,
                    num_heads=configs.n_heads,
                    flash_attention=configs.flash_attention,
                    output_attention=configs.output_attention
                ),
                d_model=configs.d_model,
                n_heads=configs.n_heads
            ),
            Intra_Agg = HyperGraphAggregateBlock(
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                num_heads=configs.n_heads,
                Learned_Q=True,
                edge_num=configs.edge_num,
                activation=configs.activation,
            ),
            Inter_Agg = HyperGraphAggregateBlock(
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                num_heads=configs.n_heads,
                activation=configs.activation,
            ),
            HyperEdgeToVertex=AttentionLayer(
                attention=HyperEdgeToVertex(
                    d_model=configs.d_model,
                    num_heads=configs.n_heads,
                    flash_attention=configs.flash_attention,
                    output_attention=configs.output_attention
                ),
                d_model=configs.d_model,
                n_heads=configs.n_heads
            ),
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation
        ) for i in range(configs.e_layers)]
        self.global_linear = nn.Linear(configs.seq_len,configs.d_model)
        self.HGblock = nn.ModuleList(self.blocks)    
        if configs.task_name == 'forecast':
            if not configs.nonautoregressive:
                self.head = nn.Linear(configs.d_model,configs.output_token_len)
            else:
                self.head = nn.Linear(configs.d_model*(configs.seq_len//configs.input_token_len),configs.test_pred_len)
        elif configs.task_name == 'imputation':
            self.head = nn.Linear(configs.d_model,configs.output_token_len)
        else:
            pass
        self.use_norm = configs.use_norm

    def forecast(self, x, x_mark, y_mark):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        B, _, C = x.shape
        # [B, C, L]
        x = x.permute(0, 2, 1)
        x_global = self.global_linear(x)
        # [B, C, N, P]
        x = x.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x.shape[2]
        # [B, C, N, D]
        embed_out = self.embedding(x)
        # [B, C * N, D]
        embed_out = embed_out.reshape(B, C * N, -1)

        for block in self.HGblock:
            embed_out,attn,_ = block(x_local=embed_out,x_global=x_global,n_vars=C,n_tokens=N)
        if not self.configs.nonautoregressive:
            # [B, C * N, P]
            dec_out = self.head(embed_out)
            # [B, C, N * P]
            dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)
        else:
            # [B,C*N,E]
            embed_out = embed_out.reshape(B, C, N, -1).reshape(B,C,-1)
            dec_out = self.head(embed_out)

        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out,0

    def imputation(self, x,mask):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        B, _, C = x.shape
        #对x进行掩码
        x = x.masked_fill(mask==0,0)
        # [B, C, L]
        x = x.permute(0, 2, 1)
        x_global = self.global_linear(x)
        # [B, C, N, P]
        x = x.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x.shape[2]
        # [B, C, N, D]
        embed_out = self.embedding(x)
        # [B, C * N, D]
        embed_out = embed_out.reshape(B, C * N, -1)
        L_aux = 0
        for block in self.HGblock:
            embed_out,attn,l_aux = block(x_local=embed_out,x_global=x_global,n_vars=C,n_tokens=N)
            L_aux += l_aux
        dec_out = self.head(embed_out)
        # [B, C, N * P]
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out,0
    

    def forward(self, x, x_mark,y, y_mark, mask):
        if self.configs.task_name == 'forecast':
            return self.forecast(x,x_mark,y_mark)
        elif self.configs.task_name == 'imputation':
            return self.imputation(x,mask)


