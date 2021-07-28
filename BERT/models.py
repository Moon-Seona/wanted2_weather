import torch
import torch.nn as nn
from transformers import GPT2Model

class BASE(nn.Module):
    def __init__(self, name_num, subject_num, max_len, factor_num):
        super(BASE, self).__init__()

        self.embed_name = nn.Embedding(name_num, factor_num)
        self.embed_subject = nn.Embedding(max_len, subject_num)
        self.embed_subject2 = nn.Linear(subject_num, factor_num)
        self.name_bias = nn.Parameter(torch.zeros((name_num,)))
        self.subject_bias = nn.Parameter(torch.zeros((subject_num,)))
        
        nn.init.normal_(self.embed_name.weight, std=0.01)
        nn.init.normal_(self.embed_subject.weight, std=0.01)
        
        self.softmax = nn.Softmax()
        
    def forward(self, subject, name, idx):
        
        print(subject)
        print(name)
        print(idx)
        

        name = self.embed_name(name)
        subject = self.embed_subject(subject)
        subject = self.embed_subject2(subject)
        prediction_i = (name * subject).sum(dim=-1) #+ item_i_bias
        prediction_i = self.softmax(prediction_i)
        
        return prediction_i

class for_puregpt(nn.Module) :
    def __init__(self, model_config, hidden, labelnum):
        super(for_puregpt, self) .__init__()

        self.model = GPT2Model.from_pretrained('skt/kogpt2-base-v2', config=model_config) #.to(device)

        self.linear = nn.Linear(hidden, labelnum)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, batch):

        gpt = self.model(**batch, output_hidden_states=True)
        gpt = gpt.to_tuple()[0]
        gpt = gpt[:,-1,:].squeeze()

        gpt = self.linear(gpt)
        gpt = self.softmax(gpt)

        return gpt