import torch
from torch import nn
from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention

class Model(nn.Module):
    """
    Unified Training of Universal Time Series Forecasting Transformers (ICML 2024)

    Paper: https://arxiv.org/abs/2402.02592
    
    GitHub: https://github.com/SalesforceAIResearch/uni2ts
    
    Citation: @inproceedings{woo2024moirai,
        title={Unified Training of Universal Time Series Forecasting Transformers},
        author={Woo, Gerald and Liu, Chenghao and Kumar, Akshat and Xiong, Caiming and Savarese, Silvio and Sahoo, Doyen},
        booktitle={Forty-first International Conference on Machine Learning},
        year={2024}
    }
    """
    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        self.use_norm = configs.use_norm
        self.pred_len = configs.test_pred_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        self.encoder = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(False, attention_dropout=configs.dropout, 
                                      output_attention=False, d_model=configs.d_model, 
                                      num_heads=configs.n_heads), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head = nn.Linear(configs.d_model, configs.input_token_len)

    def forecast(self, x, x_mark, y_mark):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        
        B, _, C = x.shape
        padding = torch.zeros(B, self.input_token_len, C).to(x.device)
        x = torch.cat([x, padding], dim=1)
        # [B, C, L]
        x = x.permute(0, 2, 1)
        # [B, C, N, P]
        x = x.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x.shape[2]
        # [B, C, N, D]
        enc_out = self.embedding(x)
        # [B, C * N, D]
        enc_out = enc_out.reshape(B, C * N, -1)
        enc_out, attns = self.encoder(enc_out, n_vars=C, n_tokens=N)
        dec_out = self.head(enc_out)
        # [B, C, N * P]
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out[:, -self.pred_len:, :]
        if self.use_norm:
            dec_out = dec_out * stdev + means
        return dec_out

    def forward(self, x, x_mark, y_mark):
        return self.forecast(x, x_mark, y_mark)