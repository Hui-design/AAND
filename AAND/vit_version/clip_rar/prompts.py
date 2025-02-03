import torch
import torch.nn as nn
import clip
import pdb

# class Prompts(nn.Module):
#     def __init__(self, initials=None, clip_model=None, len_text=None, pretrained=None):
#         super(Prompts,self).__init__()
#         print("The initial prompts are:",initials)
#         # tokenized, embedding
#         self.tokenized_prompts = clip.tokenize(initials)
#         embedding = clip_model.token_embedding(self.tokenized_prompts.cuda())
#         self.prefix, self.suffix = embedding[:, :1, :], embedding[:, 1+len_text:, :]
#         # learnable embedding
#         ctx_vectors = torch.empty(2, len_text, 768, dtype=embedding.dtype) 
#         nn.init.normal_(ctx_vectors, std=0.02)
#         self.embedding_learn = nn.Parameter(ctx_vectors)
        
#         # state_dict = torch.load(pretrained)
#         # self.embedding_prompt = state_dict['embedding_prompt'].cuda()
        
#     def forward(self, clip_model):
#         # pdb.set_trace()
#         embedding_prompt = torch.cat((self.prefix, self.embedding_learn, self.suffix), dim=1)
#         text_features = clip_model.encode_text(embedding_prompt, self.tokenized_prompts)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         return text_features


class Prompts(nn.Module):
    def __init__(self, initials=None, clip_model=None, len_text=None, pretrained=None, device='cuda'):
        super(Prompts,self).__init__()
        # initials[0] += ' normal person'
        # initials[1] += ' abnormal person'
        print("The initial prompts are:",initials)
        # tokenized, embedding
        # pdb.set_trace()
        self.tokenized_prompts = clip.tokenize(initials).to(device)
        self.model = clip_model
        embedding = self.model.token_embedding(self.tokenized_prompts).to(device)
        self.prefix, self.suffix = embedding[:, :1, :], embedding[:, 1+len_text:, :]
        self.device = device
        # self.prefix.requires_grad = False
        # self.suffix.requires_grad = False
        if pretrained is None:
            ctx_vectors = torch.zeros_like(embedding[:, 1:1+len_text, :])
            nn.init.normal_(ctx_vectors, std=0.02)
            self.embedding_prompt = nn.Parameter(ctx_vectors.requires_grad_()).to(device)
            # self.embedding_prompt = nn.Parameter(embedding[:, 1:1+len_text, :].requires_grad_()).to(device)
            # self.embedding_prompt = nn.ParameterList([nn.Parameter(torch.zeros(2,768).requires_grad_())]).to(device)
            # self.embedding_prompt = nn.Parameter(torch.randn(2,512).requires_grad_())
        else:
            state_dict = torch.load(pretrained)['embedding_prompt']
            self.embedding_prompt = state_dict.to(self.device)
        
        # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        #     nn.init.normal_(ctx_vectors, std=0.02)
        # text_features = self.embedding_prompt
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    def forward(self):
        pdb.set_trace()
        self.embedding_prompt_cat = torch.cat((self.prefix, self.embedding_prompt, self.suffix), dim=1)
        text_features = self.model.encode_text(self.embedding_prompt_cat, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features



class Linear(nn.Module):
    def __init__(self, in_dim=512, out_dim=512):
        super(Linear,self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        # self.mlp =  nn.Sequential(nn.Linear(in_dim, out_dim),
        #                         nn.ReLU(),
        #                         nn.Linear(in_dim, out_dim))

        
    def forward(self, x):
        return self.mlp(x)