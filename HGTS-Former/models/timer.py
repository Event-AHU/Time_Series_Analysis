import torch
from torch import nn
from layers.Transformer_EncDec import DecoderOnly, DecoderOnlyLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding


class Model(nn.Module):
    """
    Timer: Generative Pre-trained Transformers Are Large Time Series Models (ICML 2024)

    Paper: https://arxiv.org/abs/2402.02368
    
    GitHub: https://github.com/thuml/Large-Time-Series-Model
    
    Citation: @inproceedings{liutimer,
        title={Timer: Generative Pre-trained Transformers Are Large Time Series Models},
        author={Liu, Yong and Zhang, Haoran and Li, Chenyu and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
        booktitle={Forty-first International Conference on Machine Learning}
    }
    """
    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model, bias=False)
        self.position_embedding = PositionalEmbedding(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.blocks = DecoderOnly(
            [
                DecoderOnlyLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, 
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head = nn.Linear(configs.d_model, configs.output_token_len)
        self.use_norm = configs.use_norm

    def forecast(self, x, x_mark, y_mark):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        # [B, L, C]
        B, _, C = x.shape
        # [B, C, L]
        x = x.permute(0, 2, 1)
        # [B, C, N, P]
        x = x.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x.shape[2]
        # [B * C, N, P]
        x = x.reshape(B * C, N, -1)
        # [B * C, N, D]
        embed_out = self.embedding(x) + self.position_embedding(x)
        embed_out = self.dropout(embed_out)
        embed_out, attns = self.blocks(embed_out)
        # [B * C, N, P]
        dec_out = self.head(embed_out)
        # [B, C, L]
        dec_out = dec_out.reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * stdev + means
        return dec_out,0

    def forward(self, x, x_mark, y,y_mark,mask):
        return self.forecast(x, x_mark, y_mark)
