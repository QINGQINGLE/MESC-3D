from torch import nn
# import math
from .clip_utils import *
from torch.nn.parameter import Parameter
import math
from torch.nn import Dropout
from functools import reduce
from operator import mul
class CLIPTextEncoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=512,
                 out_dim=256,
                 patch_size = 16,
                 pretrained=None, **kwargs):
        super().__init__()
        self.layers = transformer_layers
        self.total_d_layer = transformer_layers-1
        self.pretrained = pretrained
        self.out_indices = [3, 5, 7, 11]
        self.num_tokens = 100
        self.prompt_dim = embed_dim
        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self._init_prompt(patch_size, self.num_tokens, self.prompt_dim, self.total_d_layer)
    
    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  

        if total_d_layer >= 0:
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim)) 
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if total_d_layer > 0:  # noqa
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

        else: # total_d_layer < 0
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(abs(total_d_layer), num_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length + self.num_tokens, self.context_length+self.num_tokens)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward_deep_prompt(self, embedding_output, features, out_last=False):
        B = embedding_output.shape[1] # batch_size 
        for i in range(self.layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2) # seems like the middle layer's dpt
                hidden_states = torch.cat((
                    deep_prompt_emb,
                    hidden_states[self.num_tokens:,:,:]
                ), dim=0) # 177 B 768

                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:
                # hidden_states = torch.cat((
                #     hidden_states[:1, :, :],
                #     hidden_states[-(H*W):, :, :]
                # ), dim=0)
                hidden_states = self.transformer.resblocks[i](hidden_states)
            
            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = hidden_states.permute(1, 0, 2)[:, -77:, :].permute(0, 2, 1) # B,512,77
                    features.append(xp.contiguous())
            
            if i == (self.layers-2): #10
                before_last_feats = self.prompt_norm(hidden_states) #  1125x4x768

        encoded = self.prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features 
    def encode_token(self, token):
        x = self.token_embedding(token)
        return x
    
    def forward(self, text, token):
        # x = self.token_embedding(text)
        x = text + self.positional_embedding 
        if self.total_d_layer >=0:
            # concat prompt
            x = torch.cat((  # Deep Prompt Tuning
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(x.shape[0], -1, -1)),
                    x
                ), dim=1)# B,177,512

        x = x.permute(1, 0, 2) 
        features = []
        outs = []
        if self.total_d_layer > 0: 
            x, features = self.forward_deep_prompt(x, features)
        # x = self.transformer(x)
        x = x[-77:,:,:]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), token.argmax(dim=-1)] @ self.text_projection
        # outs.append(tuple(features))
        return x

if __name__ == '__main__':
    from tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    encoder = CLIPTextEncoder(pretrained="Path/ViT-B-16.pt")
    encoder.init_weights()
    text = 'a airplane'
    texts = tokenizer(text).unsqueeze(0)
    text_embed = encoder(texts)
    print(f'text_mebed: {text_embed.shape}')
    exclude_key = 'prompt'
    for n,m in encoder.named_parameters():
        if exclude_key not in n:
            m.requires_grad = False
        else:
            print(n)
        # if exclude_key:
        #     if isinstance(exclude_key, str):
        #         if not exclude_key in n:
        #             m.requires_grad = False
        #             print(f'False : {n}')
        #     elif isinstance(exclude_key, list):
        #         count = 0
        #         for i in range(len(exclude_key)):
        #             i_layer = str(exclude_key[i])
        #             if i_layer in n:
        #                 count += 1
        #         if count == 0:
        #             m.requires_grad = False
        #         elif count>0:
        #             print('Finetune layer in backbone:', n)
        #     else:
        #         assert AttributeError("Dont support the type of exclude_key!")
        # else:
        #     m.requires_grad = False
        #     print(f'False : {n}')
            