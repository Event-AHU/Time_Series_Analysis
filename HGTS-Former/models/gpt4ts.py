import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class Model(nn.Module):
    """
    One Fits All: Power General Time Series Analysis by Pretrained LM (NeurIPS 2023 Spotlight)

    Paper: https://arxiv.org/abs/2302.11939
    
    GitHub: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All
    
    Citation: @inproceedings{zhou2023onefitsall,
        title={{One Fits All}: Power General Time Series Analysis by Pretrained LM},
        author={Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin},
        booktitle={NeurIPS},
        year={2023}
    }
    """    
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.gpt_layers = configs.gpt_layers 
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        self.gpt2 = GPT2Model.from_pretrained(
            'gpt2', attn_implementation="eager", 
            output_hidden_states=True
        )  # loads a pretrained GPT-2 base model
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.test_pred_len)
        
        for _, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                        
    def forward(self, x, x_mark, y_mark):
        B, L, C = x.shape
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        
        x = rearrange(x, 'b l c -> b c l')
        x = self.padding_patch_layer(x)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)        
        x = rearrange(x, 'b c n p -> (b c) n p')
        
        outputs = self.in_layer(x)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        outputs = self.out_layer(outputs.reshape(B*C, -1))
        outputs = rearrange(outputs, '(b c) l -> b l c', b=B)
        
        if self.use_norm:
            outputs = outputs * stdev + means
        return outputs

