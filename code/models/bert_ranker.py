#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer

import transformers
print('transformers version:', transformers.__version__)


# In[ ]:


# class BERTRanker(nn.Module):
#     def __init__(self, max_length, num_class, pretrain_path, blank_padding=True):
#         """
#         Args:
#             max_length: max length of sentence
#             pretrain_path: path of pretrain model
#         """
#         super().__init__()
#         self.max_length = max_length
#         self.blank_padding = blank_padding
#         self.hidden_size = 768
# #         self.mask_entity = mask_entity
#         logging.info('Loading BERT pre-trained checkpoint.')
#         self.bert = BertModel.from_pretrained(pretrain_path)
        
#         self.fc = nn.Linear(self.hidden_size, num_class)

#     def forward(self, token, att_mask):
#         """
#         Args:
#             token: (B, L), index of tokens
#             att_mask: (B, L), attention mask (1 for contents and 0 for padding)
#         Return:
#             x -- (B, H), representations for sentences
#             return (B, 1) scores
#         """
#         _, x = self.bert(token, attention_mask=att_mask)
#         return self.fc(x)


# In[ ]:


# class SiameseBertRanking(nn.Module):
#     def __init__(self, max_length, num_class, pretrain_path, blank_padding=True):
#         super().__init__()
        
#         self.max_length = max_length
#         self.blank_padding = blank_padding
#         self.hidden_size = 768

#         logging.info('Loading BERT pre-trained checkpoint.')
#         self.bert = BertModel.from_pretrained(pretrain_path)
        
#         self.num_classes = num_class
# #         self.loss_func = nn.CrossEntropyLoss()
        
        
#     def forward(self, q, p, q_mask, p_mask):
#         """
#         Args:
#             token: (B, L), index of tokens
#             att_mask: (B, L), attention mask (1 for contents and 0 for padding)
#         Return:
#             return (B, 1) scores
#         """
#         _, q_hiddens = self.bert(q.unsqueeze(0), attention_mask=q_mask.unsqueeze(0)) # (1, H)
#         q_hiddens = q_hiddens.squeeze(0) # (H,)
        
#         _, p_hiddens = self.bert(p, attention_mask=p_mask) # (B, H)
        
# #         print(p_hiddens.size())
# #         print(q_hiddens.size())

#         return F.cosine_similarity(p_hiddens, q_hiddens.unsqueeze(0), dim=1).squeeze(-1) # (B,)
        
# #         return torch.mm(p_hiddens, q_hiddens.unsqueeze(-1)).squeeze(-1) # (B,)


# In[ ]:


class SiameseBertClassification(nn.Module):
    def __init__(self, max_length, num_class, pretrain_path, blank_padding=True):
        super().__init__()
        
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768

        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        
        self.num_classes = num_class
        self.output_layer = nn.Linear(self.hidden_size*4, self.num_classes)
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, q, r, q_mask, r_mask):
        """
        Args:
            q: (B, L), index of tokens
            r: (B, K, L'), index of tokens
        Return:
            return (B, L) scores
        """
        _, q_hiddens = self.bert(q, attention_mask=q_mask) # (B, H)
        
        r_flat_ = r.view(r.size(0)*r.size(1), -1) #(batch_size*num_r, r_length)
        r_mask_flat_ = r_mask.view(r_mask.size(0)*r_mask.size(1), -1) #(batch_size*num_r, r_length)
        
        _, r_hiddens = self.bert(r_flat_, attention_mask=r_mask_flat_) # (BK, H)
        
        q_hiddens_expand_ = q_hiddens.unsqueeze(1).expand(q_mask.size(0), r_mask.size(1), -1)
        q_hiddens_expand_ = q_hiddens_expand_.contiguous().view(q_mask.size(0)*r_mask.size(1), -1) # (BK, H)

        matching_state = torch.cat([q_hiddens_expand_, r_hiddens,
                                    q_hiddens_expand_ - r_hiddens,
                                    q_hiddens_expand_ * r_hiddens], dim=1) #(BK, 4H)
        
        # (batch_size,)                                                                                                 
        predict = self.output_layer(matching_state).view(r_mask.size(0), -1) #(B, K)
        
        return predict
    


# In[ ]:


class SiameseBertSASRanking(nn.Module):
    def __init__(self, max_length, num_class, pretrain_path, blank_padding=True):
        super().__init__()
        
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768

        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        
        self.num_classes = num_class
        self.output_layer = nn.Linear(self.hidden_size*4, self.num_classes)
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, i, o, i_mask, o_mask):
        """
        Args:
            i: (B, L), index of tokens
            o: (B, L'), index of tokens
        Return:
            return (B, B) scores
        """
        _, i_hiddens = self.bert(i, attention_mask=i_mask) # (B, H)
        _, o_hiddens = self.bert(o, attention_mask=o_mask) # (B, H)
        
        i_hiddens_expand_ = i_hiddens.unsqueeze(1).expand(i_mask.size(0), o_mask.size(0), -1) # (B, B, H)
        o_hiddens_expand_ = o_hiddens.unsqueeze(0).expand(i_mask.size(0), o_mask.size(0), -1) # (B, B, H)
        i_hiddens_expand_ = i_hiddens_expand_.contiguous().view(i_mask.size(0)*o_mask.size(0), -1) # (BB, H)
        o_hiddens_expand_ = o_hiddens_expand_.contiguous().view(i_mask.size(0)*o_mask.size(0), -1) # (BB, H)

        matching_state = torch.cat([i_hiddens_expand_, o_hiddens_expand_,
                                    i_hiddens_expand_ - o_hiddens_expand_,
                                    i_hiddens_expand_ * o_hiddens_expand_], dim=1) #(BB, 4H)
        
        predict = self.output_layer(matching_state).view(i_mask.size(0), -1) #(B, B)
        
#         matching_state = torch.cat([i_hiddens_expand_, o_hiddens_expand_,
#                                     i_hiddens_expand_ - o_hiddens_expand_,
#                                     i_hiddens_expand_ * o_hiddens_expand_], dim=2) #(B, B, 4H)
        
#         predict = self.output_layer(matching_state) #(B, B)
        
        return predict
    


# In[ ]:


class ConcatBertRanking(nn.Module):
    def __init__(self, max_length, num_class, pretrain_path, blank_padding=True):
        super().__init__()
        
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768

        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        
        self.num_classes = num_class
        self.output_layer = nn.Linear(self.hidden_size, self.num_classes)
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, seq, mask):
        """
        Args:
            i: (B, L), index of tokens
        Return:
            return (B, B) scores
        """
        _, hiddens = self.bert(seq, attention_mask=mask) # (B, H)
        
        predict = self.output_layer(hiddens) #(B, 1)
        
        return predict
    

