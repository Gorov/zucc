#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


class BasicNLPModel(nn.Module):
    
    def __init__(self, words_lookup, args):
        super(BasicNLPModel, self).__init__()

        self.fine_tuning = args.fine_tuning
        self.vocab_size, self.embedding_dim = words_lookup.shape        
        
        # create a embedding layer
        self.embedding_layer = self._create_embedding_layer(words_lookup)
        
        # create bias for calculating similarity
        self.bias_layer = self._create_bias_layer()
                
    def _create_embedding_layer(self, words_lookup):
        embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embedding_layer.weight.data = torch.from_numpy(words_lookup)
        embedding_layer.weight.requires_grad = self.fine_tuning        
        
        return embedding_layer 
        
    def _create_bias_layer(self):    
        bias_layer = nn.Embedding(self.vocab_size, 1)
        bias_layer.weight.requires_grad = True
        
        return bias_layer               
                


# In[ ]:




