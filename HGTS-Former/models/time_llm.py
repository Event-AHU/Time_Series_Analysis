from math import sqrt
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer
from layers.Embed import PatchEmbedding

class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Time-LLM: Time Series Forecasting by Reprogramming Large Language Models (ICLR 2024)

    Paper: https://arxiv.org/abs/2310.01728
    
    GitHub: https://github.com/KimMeen/Time-LLM
    
    Citation: @inproceedings{Jin2024Time-LLM,
        title={Time Series Forecasting by Reprogramming Large Language Models},
        author={Ming Jin and Shiyu Wang and Lintao Ma and Zhixuan Chu and James Y.Zhang and Xiaoming Shi and Pin-Yu Chen and Yuxuan Liang and Yuan-Fang Li and Shirui Pan and Qingsong Wen},
        booktitle={International Conference on Learning Representations},
        year={2024}
    }
    Note: The implementation of Time-LLM based on (https://github.com/KimMeen/Time-LLM)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.test_pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.num_tokens = configs.ts_vocab_size # usually smaller than vocab_size in llm
        self.use_norm = configs.use_norm
        
        self.patch_len = configs.input_token_len
        self.stride = configs.stride
        self.domain_des = configs.domain_des # domain decomposition, recommend that set different description for different datasets
        
        if configs.llm_model == 'LLAMA':
            self.d_llm = 4096
        elif configs.llm_model == 'GPT2':
            self.d_llm = 768
        elif configs.llm_model == 'BERT':
            self.d_llm = 768
        else:
            raise Exception('LLM model is not defined')
        
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums # dim of hidden state after flatten
        
        self.top_k = 5 # number of lags
        
        self._get_model_and_tokenizer(configs.llm_model, configs.llm_layers)
        
        self._get_llm_pad_token()
        
        # freeze llm model, only need to train the reprogramming layer
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, self.stride, configs.dropout)

        # source embedding  = mapping_layer(word embedding)
        self.word_embeddings = self.llm_model.get_input_embeddings().weight #[V H]
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.output_projection = FlattenHead(self.head_nf, self.pred_len, head_dropout=configs.dropout)

    def _get_model_and_tokenizer(self, model_name, layers):
        print("> loading model: ", model_name)
        # you can also load model from local path
        if model_name == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_model = LlamaModel.from_pretrained('huggyllama/llama-7b',config=self.llama_config)
            self.tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
        elif model_name == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained('openai-community/gpt2',config=self.gpt2_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        elif model_name == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            self.llm_model = BertModel.from_pretrained('google-bert/bert-base-uncased',config=self.bert_config)
            self.tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
        else:
            raise Exception('LLM model is not defined')
        print("> loading model done")
    
    def _get_llm_pad_token(self):
        # prepare pad token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

    def _get_prompt(self, x_enc):
        # provide statistics info for prompt
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.domain_des}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)
        
        return prompt
        
    def forecast(self, x_enc, x_mark_enc, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        B, T, N = x_enc.shape # [B L M]
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1) # [B*M L 1]
        
        # get prompt embedding
        prompt = self._get_prompt(x_enc)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # [B*M prompt_token d_llm]

        # do patching
        x_enc = x_enc.reshape(B, N, T).contiguous() # [B M L]
        enc_out, n_vars = self.patch_embedding(x_enc) # [B*M N D]
        
        # source embedding = mapping_layer(word embedding)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # [num_token H]
        
        # reprogram to align the time series and natural language 
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings) # [B*M N d_llm]
        
        # concat prompt and time series and fed to LLM
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1) # [B*M prompt_token+N d_llm]
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous() # [B M d_ff prompt_token+N]
        # delete prompt tokens
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:]) # [B M L]
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    
        return dec_out
        
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def forward(self, x_enc, x_mark_enc, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
