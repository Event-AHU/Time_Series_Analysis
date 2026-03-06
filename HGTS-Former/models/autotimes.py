import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import LlamaForCausalLM, OPTForCausalLM
from layers.MLP import AutoTimesMLP

class Model(nn.Module):
    """
    AutoTimes: Autoregressive Time series Forecasters via Large Language Models (NeurIPS 2024)

    Paper: https://arxiv.org/abs/2402.02370
    
    GitHub: https://github.com/thuml/AutoTimes
    
    Citation: @inproceedings{Liu2024autotimes,
        title={AutoTimes: Autoregressive Time series Forecasters via Large Language Models},
        author={Yong Liu and Guo Qin and Xiangdong Huang and Jianmin Wang and Mingsheng Long},
        booktitle={Neural Information Processing Systems},
        year={2024}
    }
    Note: This implementation is a simplified version of  https://github.com/thuml/AutoTimes
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.input_token_len
        self.model_name = configs.llm_model
        self.mlp_hidden_dim = configs.d_model
        self.mlp_layers = configs.e_layers
        self.use_norm = configs.use_norm
        
        # load inner model
        self._get_inner_model(self.model_name)
            
        # freeze the inner model only need to train tokenizer and detokenizer
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        
        # tokenizer and detokenizer
        if self.mlp_layers == 0:
            if not configs.ddp or (configs.ddp and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim)
            self.decoder = nn.Linear(self.hidden_dim, self.token_len)
        else:
            if not configs.ddp or (configs.ddp and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = AutoTimesMLP(self.token_len, self.hidden_dim, 
                            self.mlp_hidden_dim, self.mlp_layers, 
                            configs.dropout, configs.activation)
            self.decoder = AutoTimesMLP(self.hidden_dim, self.token_len,
                            self.mlp_hidden_dim, self.mlp_layers,
                            configs.dropout, configs.activation) 

    def _get_inner_model(self, model_name):
        """
            !!! you can also load model locally or load your own model
        """
        print("> loading model: ", model_name)
        if model_name == "OPT":
            self.model = OPTForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16)
            self.model.model.decoder.project_in = None
            self.model.model.decoder.project_out = None
            self.hidden_dim = 2048
        elif model_name == "LLAMA":
            self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b", torch_dtype=torch.float32)
            self.hidden_dim = 4096
        elif model_name == "GPT2":
            self.model = GPT2Model.from_pretrained("openai-community/gpt2") 
            self.hidden_dim = 768
        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        print("> loading model done")
        
    def forecast(self, x_enc, x_mark_enc, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()    
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        bs, _, n_vars = x_enc.shape 
        x_enc = x_enc.permute(0, 2, 1) # [B M L]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        
        # tokenizer
        patch_tokens = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len) # [B*M N P]
        times_embeds = self.encoder(patch_tokens) # [B*M N H]
        
        outputs = self.model(inputs_embeds=times_embeds).last_hidden_state # [B*M N H]
        
        # detokenize
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))

        return dec_out

    def forward(self, x_enc, stamp_embeds, x_mark_dec):
        return self.forecast(x_enc, stamp_embeds, x_mark_dec)