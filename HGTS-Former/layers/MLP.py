import torch
import torch.nn as nn
import torch.nn.functional as F

class TTMGatedLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class TTMMLP(nn.Module):
    def __init__(self, in_features, out_features, factor, dropout):
        """
            factor: expansion factor for the hidden layer (usually use 2~5), in our implementation, we default it to 2
        """
        super().__init__()
        num_hidden = in_features * factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor):
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class TTMMixerBlock(nn.Module):
    def __init__(self, d_model, features, mode, dropout):
        """
            mode: mix different dimensions of input tensor based on different mode, including "patch", "feature", "channel"
        """
        super().__init__()

        self.mode = mode

        self.norm = nn.LayerNorm(d_model)

        self.mlp = TTMMLP(
            in_features=features,
            out_features=features,
            factor=2,
            dropout=dropout,
        )

        self.gating_block = TTMGatedLayer(in_size=features, out_size=features)

    def forward(self, x):
        residual = x  # [B M N P]
        x = self.norm(x)

        assert self.mode in ["patch", "feature", "channel"]

        # transpose the input tensor based on the mode so that mix the target dimension in the last dimension
        if self.mode == "patch":
            # when mode is "patch", mix the patches in the last dimension
            x = x.permute(0, 1, 3, 2)  # [B M P N]
        elif self.mode == "channel":
            # when mode is "channel", mix the channels in the last dimension
            x = x.permute(0, 3, 2, 1)  # [B P N M]
        else:
            # when mode is "feature", mix the features in the last dimension
            pass

        x = self.mlp(x)
        x = self.gating_block(x)

        # transpose the input tensor back to the original shape
        if self.mode == "patch":
            x = x.permute(0, 1, 3, 2)  # [B M N P]
        elif self.mode == "channel":
            x = x.permute(0, 3, 2, 1)  # [B M N P]
        else:
            pass

        out = x + residual
        return out


class TTMLayer(nn.Module):
    def __init__(self, d_model, num_patches, n_vars, mode, dropout):
        """
            mode: determines how to process the channels
        """
        super().__init__()

        if num_patches > 1:
            self.patch_mixer = TTMMixerBlock(
                d_model=d_model, features=num_patches, mode="patch", dropout=dropout
            )

        self.feature_mixer = TTMMixerBlock(
            d_model=d_model, features=d_model, mode="feature", dropout=dropout
        )

        self.mode = mode
        self.num_patches = num_patches
        if self.mode == "mix_channel":
            # when mode is "mix_channel", mix the channels in addition to the patches mixer and features mixer
            self.channel_feature_mixer = TTMMixerBlock(
                d_model=d_model, features=n_vars, mode="channel", dropout=dropout
            )

    def forward(self, x):
        if self.mode == "mix_channel":
            # when mode is "mix_channel", mix the channels in addition to the patches mixer and features mixer
            x = self.channel_feature_mixer(x)  # [B M N P]

        if self.num_patches > 1:
            x = self.patch_mixer(x)  # [B M N P]

        x = self.feature_mixer(x)  # [B M N P]

        return x

class AutoTimesMLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, f_in, f_out, hidden_dim=256, hidden_layers=2, dropout=0.1, activation='tanh'): 
        super(AutoTimesMLP, self).__init__()
        self.f_in = f_in
        
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y


class HyperEdgeMLP(nn.Module):
    def __init__(self,d_model, d_ff=None,activation="relu",edge_num=5,temperature=0.5,hard=True,topk=2):
        super().__init__()
        d_ff = d_ff or d_model*4
        assert edge_num>topk,"edge_num must be greater than topk"
        self.activation = nn.ReLU if activation=='relu' else nn.GELU
        # 超边数量
        self.edge_num=edge_num
        # gumbel_softmax温度参数
        self.temperature = temperature
        # 是否离散
        self.hard = hard
        # 每个顶点属于属于那些边
        self.topk = topk
        # 用于度矩阵掩码
        self.mask = torch.eye(self.edge_num).to('cuda')
        # 模式聚类
        self.MLP = nn.Sequential(
            nn.Linear(d_model,d_ff),
            self.activation(),
            nn.Linear(d_ff,edge_num)
        )
    
    def forward(self,x):
        B,L,E = x.shape
        logits = self.MLP(x)
        # 超图邻接矩阵
        E = self.gumbel_softmax(logits=logits,temperature=self.temperature,hard=self.hard,topk=self.topk)
        # 度矩阵
        D = (E.permute(0,2,1) @ E) * self.mask
        return E,D

    def gumbel_softmax(self,logits, temperature=1.0, hard=True, eps=1e-5,topk=2):
    
        # 1. 生成 Gumbel 噪声
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)

        factor = 4
        logits = logits * 4
        confidence = F.sigmoid(logits)
        # 2. 添加噪声并计算 Gumbel-Softmax
        y = logits + gumbel_noise
        y_soft = F.softmax(y / temperature, dim=-1)
        topk_probs, topk_indices = torch.topk(y_soft, topk, dim=-1)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, topk_indices, 1)

        if not hard:
            return y_soft
        return y_hard - y_soft.detach() + y_soft

        # factor = 4
        # logits = logits * 4
        # confidence = F.sigmoid(logits)
        # E = torch.zeros_like(confidence)
        # mask = (confidence > 0.5).detach()
        # E = E.masked_fill(mask,1)
        # return E - confidence.detach() + confidence

if __name__=='__main__':
    Hyper = HyperEdgeMLP(512,2048,edge_num=5)
    x = torch.rand((2,4,512),requires_grad=True)
    E,D = Hyper(x)
