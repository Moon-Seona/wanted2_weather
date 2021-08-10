import torch
import torch.nn as nn
from tokenization_kobert import KoBertTokenizer
from transformers import GPT2Model, BertModel, DistilBertModel
from kobert_transformers import get_distilkobert_model

# class BASE(nn.Module):
#     def __init__(self, name_num, subject_num, max_len, factor_num):
#         super(BASE, self).__init__()

#         self.embed_name = nn.Embedding(name_num, factor_num)
#         self.embed_subject = nn.Embedding(max_len, subject_num)
#         self.embed_subject2 = nn.Linear(subject_num, factor_num)
#         self.name_bias = nn.Parameter(torch.zeros((name_num,)))
#         self.subject_bias = nn.Parameter(torch.zeros((subject_num,)))

#         nn.init.normal_(self.embed_name.weight, std=0.01)
#         nn.init.normal_(self.embed_subject.weight, std=0.01)

#         self.softmax = nn.Softmax()

#     def forward(self, subject, name, idx):
#         print(subject)
#         print(name)
#         print(idx)

#         name = self.embed_name(name)
#         subject = self.embed_subject(subject)
#         subject = self.embed_subject2(subject)
#         prediction_i = (name * subject).sum(dim=-1)  # + item_i_bias
#         prediction_i = self.softmax(prediction_i)

#         return prediction_i


# class pure_bert(nn.Module):
#     def __init__(self, hidden, labelnum):
#         super(pure_bert, self).__init__()

#         self.model = get_distilkobert_model()
#         self.linear = nn.Linear(hidden, labelnum)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, batch):
#         bert = self.model(**batch)
#         print(type(bert))
#         print(bert.shape)  #batch, seq, hidden 48,24,768
#         dsaf

#         # tmp = torch.sum(gpt, dim=-1)
#         # for i in tmp:
#         #     print(len(i))
#         gpt = gpt[:, -1, :].squeeze()


#         gpt = self.linear(gpt)
#         gpt = self.softmax(gpt)

#         return gpt

# class for_puregpt(nn.Module):
#     def __init__(self, model_config, hidden, labelnum):
#         super(for_puregpt, self).__init__()

#         self.model1 = GPT2Model.from_pretrained('skt/kogpt2-base-v2', config=model_config)
#         #self.model2 = GPT2Model.from_pretrained('gpt2', config=model_config)
#         self.linear1 = nn.Linear(hidden, labelnum)
#         #self.linear2 = nn.Linear(hidden, labelnum)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, batch, lengths):
#         gpt1 = self.model1(**batch).last_hidden_state
#         #gpt2 = self.model(**batch).last_hidden_state

#         new_gpt1 = []
#         for batch_idx in range(len(gpt1)):
#             new_gpt1.append(gpt1[batch_idx, lengths[batch_idx]-1, :])
#         new_gpt1 = torch.stack(new_gpt1)
#         #
#         # new_gpt2 = []
#         # for batch_idx in range(len(gpt2)):
#         #     new_gpt2.append(gpt2[batch_idx, lengths[batch_idx] - 1, :])
#         # new_gpt2 = torch.stack(new_gpt2)

#         gpt = self.linear1(new_gpt1)
#         #gpt2 = self.linear2(new_gpt2)
#         #gpt = torch.sum()
#         gpt = self.softmax(gpt)

#         return gpt

# class BERTClassifier(nn.Module):
#     def __init__(self, bert, hidden_size=768, num_classes=46, dr_rate=0.5):
#         super(BERTClassifier,self).__init__()
#         self.dr_rate = dr_rate
#         self.bert = bert
#         self.classifier = nn.Linear(hidden_size, num_classes)


#     def forward(self, token_ids, valid_length, segment_ids):
#         attention_mask = self.gen_attention_mask(token_ids, valid_length)

#         _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float())

#         if self.dr_rate:
#             pooler = self.dropout(pooler)
#         return self.classifier(pooler)


class Bert_base(nn.Module):
    def __init__(self, hidden, labelnum):
        super(Bert_base, self).__init__()
        self.Bert = BertModel.from_pretrained('monologg/kobert')
        self.Linear = nn.Linear(hidden, labelnum)
        self.Softmax = nn.Softmax(dim=-1)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings # batch x seq x hidden
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # batch x seq x hidden
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) # batch x hidden
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # batch x hidden
        return sum_embeddings / sum_mask
    
    def max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings # batch x seq x hidden
        return torch.max((token_embeddings * attention_mask.unsqueeze(-1)), axis=1)
    
    def forward(self, **batch):
        # 1. text가 들어옴
        # 2. tokenizer가 임베딩함 -> 여기까지 되어서 들어옴
        # 3. 임베딩이 끝난 text를 Bert에 넣음
        # 4. Bert의 output을 mean pooling 후 linear layer에 넣음
        # 5. 그 결과에 softmax
        
        # input_ids = batch X seq, tensor 안에 list
        # token_type_ids = batch X seq
        # attention_mask = batch X seq
        
        # return 값은 batch X label

        text_output = self.Bert(**batch)
        text_embedding = self.mean_pooling(text_output, batch['attention_mask'])
        text_embedding = self.Linear(text_embedding)
        text_softmax = self.Softmax(text_embedding) # batch X label
        return text_softmax # batch X label