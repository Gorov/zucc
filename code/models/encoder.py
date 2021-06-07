#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
# from models import CnnModel, RnnModel
import numpy as np

import math
from torch.autograd import Variable  


# In[ ]:


class CnnModel(nn.Module):
    
    def __init__(self, args):
        """
        args.hidden_dim -- dimension of filters
        args.embedding_dim -- dimension of word embeddings
        args.kernel_size -- kernel size of the conv1d
        args.layer_num -- number of CNN layers
        """
        super(CnnModel, self).__init__()

        self.args = args
        if args.kernel_size % 2 == 0:
            raise ValueError("args.kernel_size should be an odd number")
            
        self.conv_layers = nn.Sequential()
        for i in range(args.layer_num):
            if i == 0:
                input_dim = args.embedding_dim
            else:
                input_dim = args.hidden_dim
            self.conv_layers.add_module('conv_layer{:d}'.format(i), nn.Conv1d(in_channels=input_dim, 
                                                  out_channels=args.hidden_dim, kernel_size=args.kernel_size,
                                                                             padding=(args.kernel_size-1)/2))
            self.conv_layers.add_module('relu{:d}'.format(i), nn.ReLU())
        
    def forward(self, embeddings):
        """
        Given input embeddings in shape of (batch_size, sequence_length, embedding_dim) generate a 
        sentence embedding tensor (batch_size, sequence_length, hidden_dim)
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length)       
        """
        embeddings_ = embeddings.transpose(1, 2) #(batch_size, embedding_dim, sequence_length)
        hiddens = self.conv_layers(embeddings_)
        return hiddens


# In[ ]:


class RnnModel(nn.Module):

    def __init__(self, args):
        """
        args.hidden_dim -- dimension of filters
        args.embedding_dim -- dimension of word embeddings
        args.layer_num -- number of RNN layers   
        args.cell_type -- type of RNN cells, GRU or LSTM
        """
        super(RnnModel, self).__init__()
        
        self.args = args
 
        if args.cell_type == 'GRU':
            self.rnn_layer = nn.GRU(input_size=args.embedding_dim, 
                                    hidden_size=args.hidden_dim//2, 
                                    num_layers=args.layer_num, bidirectional=True)
        elif args.cell_type == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size=args.embedding_dim, 
                                     hidden_size=args.hidden_dim//2, 
                                     num_layers=args.layer_num, bidirectional=True)
    
    def forward(self, embeddings, mask=None):
        """
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
            mask -- a float tensor of masks, (batch_size, length)
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length)
        """
        embeddings_ = embeddings.transpose(0, 1) #(sequence_length, batch_size, embedding_dim)
        
        if mask is not None:
            seq_lengths = list(torch.sum(mask, dim=1).cpu().data.numpy())
            seq_lengths = [int(x) for x in seq_lengths]
#             seq_lengths = map(int, seq_lengths)
            inputs_ = torch.nn.utils.rnn.pack_padded_sequence(embeddings_, seq_lengths)
        else:
            inputs_ = embeddings_
        
        hidden, _ = self.rnn_layer(inputs_) #(sequence_length, batch_size, hidden_dim (* 2 if bidirectional))
        
        if mask is not None:
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden) #(length, batch_size, hidden_dim)
        
        return hidden.permute(1, 2, 0) #(batch_size, hidden_dim, sequence_length)


# In[ ]:


class Encoder(nn.Module):
    
    def __init__(self, args):
        """
        Inputs:
        args.model_type -- "CNN" or "RNN"
        if use CNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.kernel_size -- kernel size of the conv1d
            args.layer_num -- number of CNN layers
        if use RNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.layer_num -- number of RNN layers   
            args.cell_type -- type of RNN cells, "GRU" or "LSTM"
        """
        super(Encoder, self).__init__()
        
        self.args = args
        
        if args.model_type == "CNN":
            self.encoder_model = CnnModel(args)
        elif args.model_type == "RNN":
            self.encoder_model = RnnModel(args)
                
    def forward(self, x, z, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        regression value of each input
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
            z -- input rationale, ``binary'' mask, (batch_size, sequence_length)
        Outputs:
            output -- hidden values at all time step, shape: (batch_size, hidden_dim, sequence_length) 
        """
        masked_input = x * z.unsqueeze(-1) #(batch_size, sequence_length, embedding_dim)        
        hiddens = self.encoder_model(masked_input, mask) #(batch_size, hidden_dim, sequence_length)        
        return hiddens


# In[ ]:


class ClassificationEncoder(Encoder):
    
    def __init__(self, args):
        super(ClassificationEncoder, self).__init__(args)
        self.num_classes = args.num_classes
        self.output_layer = nn.Linear(args.hidden_dim, self.num_classes)
                
    def forward(self, x, z, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        regression value of each input
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
            z -- input rationale, ``binary'' mask, (batch_size, sequence_length)
        Outputs:
            output -- output of the regression value, a vector of size batch_size
        """
        hiddens = super(ClassificationEncoder, self).forward(x, z, mask)
        
        if mask is not None:
            neg_inf = -1.0e6
            hiddens = hiddens + (1 - mask.unsqueeze(1)) * neg_inf 
        
        max_hidden = torch.max(hiddens, -1)[0] #(batch_size, hidden_dim)        
        output = self.output_layer(max_hidden)
        
        return output


# In[ ]:


class MLPClassificationEncoder(nn.Module):
    def __init__(self, args, num_classes=2):
        """
        Inputs:
        model_type -- "Linear" or "MLP"
        if use MLP:
            hidden_dim -- dimension of filters
        """
        super(MLPClassificationEncoder, self).__init__()
        
        self.args = args
        self.num_classes = num_classes        

        self.hidden_layer = nn.Sequential()
        self.hidden_layer.add_module('linear', nn.Linear(args.embedding_dim, args.hidden_dim))
        self.hidden_layer.add_module('relu', nn.ReLU())
        self.output_layer = nn.Linear(args.hidden_dim, self.num_classes)

    def forward(self, x, z, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a
        list of embeddings
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            output -- hidden values at all time step, shape: (batch_size, hidden_dim, sequence_length)
        """
        masked_input = x * z.unsqueeze(-1)
        hiddens = self.hidden_layer(masked_input) #(batch_size, sequence_length, hidden)

        if mask is not None:
            neg_inf = -1.0e6
            hiddens = hiddens + (1 - mask.unsqueeze(-1)) * neg_inf 
        
        max_hidden = torch.max(hiddens, dim=1)[0] #(batch_size, hidden_dim)            
        output = self.output_layer(max_hidden).squeeze(-1) #(batch_size,)
        return output

