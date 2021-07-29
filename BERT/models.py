import torch
import torch.nn as nn
from transformers import GPT2Model, BertModel, DistilBertModel
from kobert_transformers import get_distilkobert_model

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
        prediction_i = (name * subject).sum(dim=-1)  # + item_i_bias
        prediction_i = self.softmax(prediction_i)

        return prediction_i


class pure_bert(nn.Module):
    def __init__(self, hidden, labelnum):
        super(pure_bert, self).__init__()

        self.model = get_distilkobert_model()
        self.linear = nn.Linear(hidden, labelnum)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch):
        bert = self.model(**batch)
        print(type(bert))
        print(bert.shape)  #batch, seq, hidden 48,24,768
        dsaf

        # tmp = torch.sum(gpt, dim=-1)
        # for i in tmp:
        #     print(len(i))
        gpt = gpt[:, -1, :].squeeze()


        gpt = self.linear(gpt)
        gpt = self.softmax(gpt)

        return gpt

class for_puregpt(nn.Module):
    def __init__(self, model_config, hidden, labelnum):
        super(for_puregpt, self).__init__()

        self.model1 = GPT2Model.from_pretrained('skt/kogpt2-base-v2', config=model_config)
        #self.model2 = GPT2Model.from_pretrained('gpt2', config=model_config)
        self.linear1 = nn.Linear(hidden, labelnum)
        #self.linear2 = nn.Linear(hidden, labelnum)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch, lengths):
        gpt1 = self.model1(**batch).last_hidden_state
        #gpt2 = self.model(**batch).last_hidden_state

        new_gpt1 = []
        for batch_idx in range(len(gpt1)):
            new_gpt1.append(gpt1[batch_idx, lengths[batch_idx]-1, :])
        new_gpt1 = torch.stack(new_gpt1)
        #
        # new_gpt2 = []
        # for batch_idx in range(len(gpt2)):
        #     new_gpt2.append(gpt2[batch_idx, lengths[batch_idx] - 1, :])
        # new_gpt2 = torch.stack(new_gpt2)

        gpt = self.linear1(new_gpt1)
        #gpt2 = self.linear2(new_gpt2)
        #gpt = torch.sum()
        gpt = self.softmax(gpt)

        return gpt
