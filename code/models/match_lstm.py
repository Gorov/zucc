#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .encoder import Encoder, ClassificationEncoder, MLPClassificationEncoder
# from utils.utils import l21_norm, continuity_loss_func, corner_detection, fused_sparsity_loss_batch
from .basic_nlp_models import BasicNLPModel
import numpy as np
import copy
import os


# In[ ]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)

        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
            sort_idx = np.argsort(-lens)
            idx_dict = {sort_idx[i_]: i_ for i_ in range(lens.shape[0])}
            revert_idx = np.array([idx_dict[i_] for i_ in range(lens.shape[0])])
            input = input[sort_idx, :]
        output = input
            
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
#             print(output.size())
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens[sort_idx], batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
#             print(output.size())
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        
        if input_lengths is not None: 
            if self.concat:
                return torch.cat(outputs, dim=2)[revert_idx,:]
            return outputs[-1][revert_idx,:]
        else:
            if self.concat:
                return torch.cat(outputs, dim=2)
            return outputs[-1]


# In[ ]:


class RelationalMatchLSTMEncoder(nn.Module):
    def __init__(self, args):
        super(RelationalMatchLSTMEncoder, self).__init__()
        # create an encoder layer
        encoder_args = copy.deepcopy(args)
        encoder_args.model_type = 'RNN'
        encoder_args.input_dim = args.embedding_dim
        encoder_args.layer_num = args.layer_num
        
        self.encoder = Encoder(encoder_args) 

        # create an output layer
        match_lstm_args = copy.deepcopy(args)
        match_lstm_args.model_type = 'RNN'
        match_lstm_args.layer_num = 1
        match_lstm_args.embedding_dim = match_lstm_args.hidden_dim*4
        match_lstm_args.hidden_dim = args.mlstm_hidden_dim
        match_lstm_args.num_classes = 1
        
        self.match_lstm = ClassificationEncoder(match_lstm_args)

        
    def _calculate_similarity_matrix(self, q_hiddens, p_hiddens):
        """
        Inputs: 
            q_hiddens -- (batch_size, hidden_dim, sequence_length_q)                                                    
            rw_hiddens -- (batch_size, hidden_dim, sequence_length_p)    
        """
       
        #(batch_size, q_length, p_length)
        similarity_matrix = torch.bmm(q_hiddens.transpose(1, 2), p_hiddens)
        
        return similarity_matrix
    
    
    def _get_r_matching_representations(self, q_hiddens, p_hiddens, similarity_matrix, mask_matrix=None):
        """                                                                                                             
        This function takes the sequences of hidden states and the word-by-word attention matrix,                       
        and returns                                                                                                     
                                                                                                                        
        Inputs:                                                                                                         
            q_hiddens -- (batch_size, hidden_dim, sequence_length_q)                                                    
            p_hiddens -- (batch_size, hidden_dim, sequence_length_p)                                                    
            similarity_matrix -- (batch_size, sequence_length_q, sequence_length_p)     
            z_matrix -- (batch_size, q_length, p_length)
                                                                                                                        
        Outputs:                                                                                                        
            q_matching_states -- (batch_size, sequence_length_q， hidden_dim * 4)                                       
        """
#         attention_softmax = F.softmax(similarity_matrix, dim=2) * z_matrix
        neg_inf = -1.0e6
        if mask_matrix is not None:
            attention_softmax = F.softmax(similarity_matrix + (1 * mask_matrix) * neg_inf, dim=2)
        else:
            attention_softmax = F.softmax(similarity_matrix, dim=2)
        
        # shape: (batch_size, sequence_length_q, hidden_dim)                                                            
        q_hiddens_tilda = torch.bmm(attention_softmax, p_hiddens.transpose(1, 2))
        q_hiddens_ = q_hiddens.transpose(1, 2)

        # shape: (batch_size, sequence_length_q, hidden_dim * 4)                                                        
        q_matching_states = torch.cat([q_hiddens_, q_hiddens_tilda,
                                      q_hiddens_ - q_hiddens_tilda,
                                      q_hiddens_ * q_hiddens_tilda], dim=2)
        return q_matching_states, attention_softmax
        
    def forward(self, q_embeddings, r_embeddings, z_q, z_rw, q_mask, r_mask):
        
        q_len = torch.sum(q_mask, dim=1).cpu().data.numpy()
        
        sort_idx = np.argsort(-q_len)
        idx_dict = {sort_idx[i_]: i_ for i_ in range(q_len.shape[0])}
        revert_idx = np.array([idx_dict[i_] for i_ in range(q_len.shape[0])])
        
        q_hiddens_sort_ = self.encoder(q_embeddings[sort_idx,:,:], z_q[sort_idx,:], q_mask[sort_idx,:]) #(batch_size, hidden_dim, q_length)
        q_hiddens = q_hiddens_sort_[revert_idx, :, :].contiguous()
#         q_hiddens = self.encoder(q_embeddings, z_q, q_mask)
        
#         print(q_hiddens.size())

        r_mask_flat_ = r_mask.view(r_mask.size(0)*r_mask.size(1), -1) #(batch_size*num_r, rw_length)
        r_embeddings_flat_ = r_embeddings.view(r_mask.size(0)*r_mask.size(1), r_mask.size(2), -1)
        
        r_len = torch.sum(r_mask_flat_, dim=1).cpu().data.numpy()
        sort_idx = np.argsort(-r_len)
        idx_dict = {sort_idx[i_]: i_ for i_ in range(r_len.shape[0])}
        revert_idx = np.array([idx_dict[i_] for i_ in range(r_len.shape[0])])
        
        r_hiddens_sort_ = self.encoder(r_embeddings_flat_[sort_idx,:,:], 
                                   z_rw.view(z_rw.size(0)*z_rw.size(1), -1)[sort_idx,:],
                                   r_mask_flat_[sort_idx,:]) #(batch*num_r, hidden_dim, rw_length)
        
        r_hiddens_ = r_hiddens_sort_[revert_idx,:,:].contiguous()
        
#         print(r_hiddens_.size())
        
#         rw_hiddens = rw_hiddens_.view(rw.size(0), rw.size(1), -1, rw.size(2)) #(batch, num_r, hidden_dim, rw_length)
        
        # expand to (batch*num_r, hidden_dim, q_length) and (batch*num_r, q_length)
#         q_hiddens_expand_ = q_hiddens.unsqueeze(1).expand(q_mask.size(0), r_mask.size(1), -1, q_mask.size(1))
#         q_hiddens_expand_ = q_hiddens_expand_.contiguous().view(q_mask.size(0)*r_mask.size(1), -1, q_mask.size(1))
#         q_mask_expand_ = q_mask.unsqueeze(1).expand(q_mask.size(0), r_mask.size(1), q_mask.size(1))
#         q_mask_expand_ = q_mask_expand_.contiguous().view(-1, q_mask.size(1))
        
        q_hiddens_expand_ = q_hiddens.unsqueeze(1).expand(q_mask.size(0), r_mask.size(1), -1, q_hiddens.size(2))
        q_hiddens_expand_ = q_hiddens_expand_.contiguous().view(q_mask.size(0)*r_mask.size(1), -1, q_hiddens.size(2))
        q_mask_expand_ = torch.ones(q_mask.size(0) * r_mask.size(1), q_hiddens.size(2)).cuda()

    
        # generate word-by-word similarity between q and p, (batch_size*num_r, rw_length, q_length)            
        similarity_matrix = self._calculate_similarity_matrix(r_hiddens_, q_hiddens_expand_)
        mask_matrix = torch.bmm(r_mask_flat_.unsqueeze(2), q_mask_expand_.unsqueeze(1))
        
        # (batch_size*num_r, rw_length， hidden_dim * 4)                                                              
        r_matching_states, _ = self._get_r_matching_representations(r_hiddens_, q_hiddens_expand_,
                                                                 similarity_matrix, mask_matrix)
        # (batch_size*num_r,)
#         predict = self.match_lstm(r_matching_states, 
#                                   z_rw.view(z_rw.size(0)*z_rw.size(1), -1),
#                                   r_mask_flat_)
#         predict = predict.view(r_mask.size(0), r_mask.size(1)) # (batch, num_r)

        predict_sort_ = self.match_lstm(r_matching_states[sort_idx,:,:], 
                                  z_rw.view(z_rw.size(0)*z_rw.size(1), -1)[sort_idx,:],
                                  r_mask_flat_[sort_idx,:])
        predict = predict_sort_[revert_idx,:].contiguous().view(r_mask.size(0), r_mask.size(1)) # (batch, num_r)
        
        return predict
        


# In[ ]:


class MatchLSTMEncoder(nn.Module):
    def __init__(self, args):
        super(RelationalMatchLSTMEncoder, self).__init__()
        # create an encoder layer
        encoder_args = copy.deepcopy(args)
        encoder_args.model_type = 'RNN'
        encoder_args.input_dim = args.embedding_dim
        encoder_args.layer_num = args.layer_num
        
        self.encoder = Encoder(encoder_args) 

        # create an output layer
        match_lstm_args = copy.deepcopy(args)
        match_lstm_args.model_type = 'RNN'
        match_lstm_args.layer_num = 1
        match_lstm_args.embedding_dim = match_lstm_args.hidden_dim*4
        match_lstm_args.hidden_dim = args.mlstm_hidden_dim
        match_lstm_args.num_classes = 1
        
        self.match_lstm = ClassificationEncoder(match_lstm_args)

        
    def _calculate_similarity_matrix(self, q_hiddens, p_hiddens):
        """
        Inputs: 
            q_hiddens -- (batch_size, hidden_dim, sequence_length_q)                                                    
            rw_hiddens -- (batch_size, hidden_dim, sequence_length_p)    
        """
       
        #(batch_size, q_length, p_length)
        similarity_matrix = torch.bmm(q_hiddens.transpose(1, 2), p_hiddens)
        
        return similarity_matrix
    
    
    def _get_matching_representations(self, q_hiddens, p_hiddens, similarity_matrix, mask_matrix=None):
        """                                                                                                             
        This function takes the sequences of hidden states and the word-by-word attention matrix,                       
        and returns                                                                                                     
                                                                                                                        
        Inputs:                                                                                                         
            q_hiddens -- (batch_size, hidden_dim, sequence_length_q)                                                    
            p_hiddens -- (batch_size, hidden_dim, sequence_length_p)                                                    
            similarity_matrix -- (batch_size, sequence_length_q, sequence_length_p)     
            z_matrix -- (batch_size, q_length, p_length)
                                                                                                                        
        Outputs:                                                                                                        
            q_matching_states -- (batch_size, sequence_length_q， hidden_dim * 4)                                       
        """
#         attention_softmax = F.softmax(similarity_matrix, dim=2) * z_matrix
        neg_inf = -1.0e6
        if mask_matrix is not None:
            attention_softmax = F.softmax(similarity_matrix + (1 * mask_matrix) * neg_inf, dim=2)
        else:
            attention_softmax = F.softmax(similarity_matrix, dim=2)
        
        # shape: (batch_size, sequence_length_q, hidden_dim)                                                            
        q_hiddens_tilda = torch.bmm(attention_softmax, p_hiddens.transpose(1, 2))
        q_hiddens_ = q_hiddens.transpose(1, 2)

        # shape: (batch_size, sequence_length_q, hidden_dim * 4)                                                        
        q_matching_states = torch.cat([q_hiddens_, q_hiddens_tilda,
                                      q_hiddens_ - q_hiddens_tilda,
                                      q_hiddens_ * q_hiddens_tilda], dim=2)
        return q_matching_states, attention_softmax
        
    def forward(self, q_embeddings, r_embeddings, z_q, z_rw, q_mask, r_mask):
        
        q_len = torch.sum(q_mask, dim=1).cpu().data.numpy()
        
        q_sort_idx = np.argsort(-q_len)
        q_idx_dict = {q_sort_idx[i_]: i_ for i_ in range(q_len.shape[0])}
        q_revert_idx = np.array([q_idx_dict[i_] for i_ in range(q_len.shape[0])])
        
        q_hiddens_sort_ = self.encoder(q_embeddings[q_sort_idx,:,:], z_q[q_sort_idx,:], 
                                       q_mask[q_sort_idx,:]) #(batch_size, hidden_dim, q_length)
        q_hiddens = q_hiddens_sort_[q_revert_idx, :, :].contiguous()
#         q_hiddens = self.encoder(q_embeddings, z_q, q_mask)
        
#         print(q_hiddens.size())

        r_mask_flat_ = r_mask.view(r_mask.size(0)*r_mask.size(1), -1) #(B1*B2, L2)
        r_embeddings_flat_ = r_embeddings.view(r_mask.size(0)*r_mask.size(1), r_mask.size(2), -1)
        
        r_len = torch.sum(r_mask_flat_, dim=1).cpu().data.numpy()
        sort_idx = np.argsort(-r_len)
        idx_dict = {sort_idx[i_]: i_ for i_ in range(r_len.shape[0])}
        revert_idx = np.array([idx_dict[i_] for i_ in range(r_len.shape[0])])
        
        r_hiddens_sort_ = self.encoder(r_embeddings_flat_[sort_idx,:,:], 
                                   z_rw.view(z_rw.size(0)*z_rw.size(1), -1)[sort_idx,:],
                                   r_mask_flat_[sort_idx,:]) #(B1*B2, H, L2)
        
        r_hiddens_ = r_hiddens_sort_[revert_idx,:,:].contiguous()
        
        q_hiddens_expand_ = q_hiddens.unsqueeze(1).expand(q_mask.size(0), r_mask.size(1), -1, q_hiddens.size(2))
        q_hiddens_expand_ = q_hiddens_expand_.contiguous().view(q_mask.size(0)*r_mask.size(1), -1, q_hiddens.size(2))
        q_mask_expand_ = torch.ones(q_mask.size(0) * r_mask.size(1), q_hiddens.size(2)).cuda() #(B1*B2, L1)

        qe_len = torch.sum(q_mask_expand_, dim=1).cpu().data.numpy()
        qe_sort_idx = np.argsort(-qe_len)
        qe_idx_dict = {qe_sort_idx[i_]: i_ for i_ in range(qe_len.shape[0])}
        qe_revert_idx = np.array([qe_idx_dict[i_] for i_ in range(qe_len.shape[0])])
    
        # generate word-by-word similarity between q and p, (B1*B2, L1, L2)
        similarity_matrix = self._calculate_similarity_matrix(q_hiddens_expand_, r_hiddens_)
        mask_matrix = torch.bmm(q_mask_expand_.unsqueeze(2), r_mask_flat_.unsqueeze(1)) #(B1*B2, L1, L2)
        
        # (B1*B2, L1， hidden_dim * 4)                                                              
        q_matching_states, _ = self._get_matching_representations(r_hiddens_, q_hiddens_expand_,
                                                                 similarity_matrix, mask_matrix)

        
        
        predict_sort_ = self.match_lstm(r_matching_states[qe_sort_idx,:,:], 
                                  z_rw.view(z_rw.size(0)*z_rw.size(1), -1)[qe_sort_idx,:],
                                  q_mask_expand_[qe_sort_idx,:])
        predict = predict_sort_[qe_revert_idx,:].contiguous().view(r_mask.size(0), r_mask.size(1)) # (B1, B2)
        
        return predict
        


# In[ ]:


class MatchLSTMRankingModel(BasicNLPModel):
    def __init__(self, words_lookup, args):
        super(MatchLSTMRankingModel, self).__init__(words_lookup, args)
        
        self.num_classes = args.num_classes
        self.hidden_dim = args.hidden_dim
        self.is_cuda = args.cuda
        self.model_type = args.model_type

        # create an encoder layer
        self.match_lstm = RelationalMatchLSTMEncoder(args)
#         self.match_lstm = MatchLSTMEncoder(args)
        
        # create a loss function 
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        
        
    def _create_embed_layer(self, vocab_size, emb_dim):
        embed_layer = nn.Embedding(vocab_size, emb_dim)
        embed_layer.weight.requires_grad = True
        return embed_layer
        
        
    def forward(self, q, r, q_mask, r_mask):
        
        q_embeddings = self.embedding_layer(q) #(batch_size, q_length, embedding_dim)
            
        rw_embeddings = self.embedding_layer(r) #(batch_size, num_r, r_length, embedding_dim)         
        
        z_q = torch.ones_like(q).float().cuda() # (batch, q_length)
        z_r = torch.ones_like(r).float().cuda() # (batch, num_r, p_length)
        
        predict = self.match_lstm(q_embeddings, rw_embeddings, z_q, z_r, q_mask, r_mask)
        
        return predict
    
    def loss(self, predict, label):
        prediction_loss = self.loss_func(predict, label) # (batch_size, )
        supervised_loss = torch.mean(prediction_loss)
                
        return supervised_loss


# In[ ]:


class MatchLSTMForwardRankingModel(BasicNLPModel):
    def __init__(self, words_lookup, args):
        super(MatchLSTMForwardRankingModel, self).__init__(words_lookup, args)
        
        self.num_classes = args.num_classes
        self.hidden_dim = args.hidden_dim
        self.is_cuda = args.cuda
        self.model_type = args.model_type

        # create an encoder layer
        self.match_lstm = RelationalMatchLSTMEncoder(args)
#         self.match_lstm = MatchLSTMEncoder(args)
        
        # create a loss function 
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        
        
    def _create_embed_layer(self, vocab_size, emb_dim):
        embed_layer = nn.Embedding(vocab_size, emb_dim)
        embed_layer.weight.requires_grad = True
        return embed_layer
        
        
    def forward(self, i, o, i_mask, o_mask):
        
        i_embeddings = self.embedding_layer(i) #(B1, L1, E)
#         print(i_embeddings.size())
            
        o_embeddings = self.embedding_layer(o) #(B2, L2, E)
        o_embeddings = o_embeddings.unsqueeze(0).expand(i.size(0), o.size(0),
                                                       o.size(1), o_embeddings.size(2)).contiguous() # (B1, B2, L2, E)
        
        z_i = torch.ones_like(i).float().cuda() # (B1, L)
        z_o = torch.ones_like(o).float().cuda() # (B2, L)
        z_o = z_o.unsqueeze(0).expand(i.size(0), o.size(0), o.size(1)).contiguous() # (B1, B2, L2)
        o_mask = o_mask.unsqueeze(0).expand(i.size(0), o.size(0), o.size(1)).contiguous() # (B1, B2, L2)
        
        predict = self.match_lstm(i_embeddings, o_embeddings, z_i, z_o, i_mask, o_mask)
        
        return predict
    
    def loss(self, predict, label):
        prediction_loss = self.loss_func(predict, label) # (B,)
        supervised_loss = torch.mean(prediction_loss)
                
        return supervised_loss


# In[ ]:


class MatchLSTMForwardDiffRankingModel(BasicNLPModel):
    def __init__(self, words_lookup, args):
        super(MatchLSTMForwardDiffRankingModel, self).__init__(words_lookup, args)
        
        self.num_classes = args.num_classes
        self.hidden_dim = args.hidden_dim
        self.is_cuda = args.cuda
        self.model_type = args.model_type
        
        self.state_linear = nn.Sequential(
                nn.Linear(config.hidden, config.hidden),
                nn.ReLU()
            )
        
        self.next_state_linear = nn.Sequential(
                nn.Linear(config.hidden, config.hidden),
                nn.ReLU()
            )

        # create an encoder layer
#         self.match_lstm = RelationalMatchLSTMEncoder(args)
        self.match_lstm = MatchLSTMEncoder(args)
        
        # create a loss function 
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        
        
    def _create_embed_layer(self, vocab_size, emb_dim):
        embed_layer = nn.Embedding(vocab_size, emb_dim)
        embed_layer.weight.requires_grad = True
        return embed_layer
        
        
    def forward(self, i, o, i_mask, o_mask):
        
        i_embeddings = self.embedding_layer(i) #(B1, L1, E)
#         print(i_embeddings.size())
            
        o_embeddings = self.embedding_layer(o) #(B2, L2, E)
        o_embeddings = o_embeddings.unsqueeze(0).expand(i.size(0), o.size(0),
                                                       o.size(1), o_embeddings.size(2)).contiguous() # (B1, B2, L2, E)
        
        z_i = torch.ones_like(i).float().cuda() # (B1, L)
        z_o = torch.ones_like(o).float().cuda() # (B2, L)
        z_o = z_o.unsqueeze(0).expand(i.size(0), o.size(0), o.size(1)).contiguous() # (B1, B2, L2)
        o_mask = o_mask.unsqueeze(0).expand(i.size(0), o.size(0), o.size(1)).contiguous() # (B1, B2, L2)
        
        predict = self.match_lstm(i_embeddings, o_embeddings, z_i, z_o, i_mask, o_mask)
        
        return predict
    
    def loss(self, predict, label):
        prediction_loss = self.loss_func(predict, label) # (B,)
        supervised_loss = torch.mean(prediction_loss)
                
        return supervised_loss


# In[ ]:


class DifferentiateMatchLSTMEncoder(RelationalMatchLSTMEncoder):
    def __init__(self, args):
        super(DifferentiateMatchLSTMEncoder, self).__init__(args)
        # create an diff layer
        match_lstm_args = copy.deepcopy(args)
        match_lstm_args.model_type = 'RNN'
        match_lstm_args.layer_num = 1
        match_lstm_args.embedding_dim = match_lstm_args.hidden_dim*4
        
        self.diff_match_lstm = Encoder(match_lstm_args)
        
#         self.linear_proj_qp = nn.Sequential(
#             nn.Linear(args.hidden_dim*4, args.hidden_dim),
#             nn.ReLU()
#         )
        
    def forward(self, q_embeddings, p_embeddings, r_embeddings, z_q, z_p, z_rw, q_mask, p_mask, r_mask):
        # q encoding
        q_len = torch.sum(q_mask, dim=1).cpu().data.numpy()
        sort_idx = np.argsort(-q_len)
        idx_dict = {sort_idx[i_]: i_ for i_ in range(q_len.shape[0])}
        revert_idx = np.array([idx_dict[i_] for i_ in range(q_len.shape[0])])
        
        q_hiddens_sort_ = self.encoder(q_embeddings[sort_idx,:,:], z_q[sort_idx,:], q_mask[sort_idx,:]) #(batch_size, hidden_dim, q_length)
        q_hiddens = q_hiddens_sort_[revert_idx, :, :].contiguous()
        
        # p encoding
        p_len = torch.sum(p_mask, dim=1).cpu().data.numpy()
        p_sort_idx = np.argsort(-p_len)
        p_idx_dict = {p_sort_idx[i_]: i_ for i_ in range(p_len.shape[0])}
        p_revert_idx = np.array([p_idx_dict[i_] for i_ in range(p_len.shape[0])])
        
        p_hiddens_sort_ = self.encoder(p_embeddings[p_sort_idx,:,:], z_p[p_sort_idx,:], p_mask[p_sort_idx,:])
        p_hiddens = p_hiddens_sort_[p_revert_idx, :, :].contiguous()

        # r encoding
        r_mask_flat_ = r_mask.view(r_mask.size(0)*r_mask.size(1), -1) #(batch_size*num_r, rw_length)
        r_embeddings_flat_ = r_embeddings.view(r_mask.size(0)*r_mask.size(1), r_mask.size(2), -1)
        
        r_len = torch.sum(r_mask_flat_, dim=1).cpu().data.numpy()
        sort_idx = np.argsort(-r_len)
        idx_dict = {sort_idx[i_]: i_ for i_ in range(r_len.shape[0])}
        revert_idx = np.array([idx_dict[i_] for i_ in range(r_len.shape[0])])
        
        r_hiddens_sort_ = self.encoder(r_embeddings_flat_[sort_idx,:,:], 
                                   z_rw.view(z_rw.size(0)*z_rw.size(1), -1)[sort_idx,:],
                                   r_mask_flat_[sort_idx,:]) #(batch*num_r, hidden_dim, rw_length)
        
        r_hiddens_ = r_hiddens_sort_[revert_idx,:,:].contiguous()
        
        # generate word-by-word similarity between q and p, (batch_size, p_length, q_length)
        pq_similarity_matrix = self._calculate_similarity_matrix(p_hiddens, q_hiddens)
        pq_mask_matrix = torch.bmm(p_mask.unsqueeze(2), q_mask.unsqueeze(1))
        
        # (batch_size, p_length, hidden_dim * 4)                                                              
        p_matching_states, _ = self._get_r_matching_representations(p_hiddens, q_hiddens,
                                                                 pq_similarity_matrix, pq_mask_matrix)
        
#         p_matching_states = self.linear_proj_qp(p_matching_states).transpose(1, 2).contiguous() # (batch_size, hidden_dim, p_length)
#         p_matching_states += p_hiddens
        p_matching_states_sort_ = self.diff_match_lstm(p_matching_states[p_sort_idx,:,:],
                                                      z_p[p_sort_idx,:], p_mask[p_sort_idx,:]) 
        p_matching_states = p_matching_states_sort_[p_revert_idx, :, :].contiguous() # (batch_size, hidden_dim, p_length)
        p_matching_states += p_hiddens
        
        # expand to (batch*num_r, hidden_dim, p_length) and (batch*num_r, p_length)
        p_matching_expand_ = p_matching_states.unsqueeze(1).expand(p_mask.size(0), r_mask.size(1), -1, p_mask.size(1))
        p_matching_expand_ = p_matching_expand_.contiguous().view(p_mask.size(0)*r_mask.size(1), -1, p_mask.size(1))
        p_mask_expand_ = p_mask.unsqueeze(1).expand(p_mask.size(0), r_mask.size(1), p_mask.size(1))
        p_mask_expand_ = p_mask_expand_.contiguous().view(-1, p_mask.size(1))

    
        # generate word-by-word similarity between q and p, (batch_size*num_r, rw_length, q_length)            
        similarity_matrix = self._calculate_similarity_matrix(r_hiddens_, p_matching_expand_)
        mask_matrix = torch.bmm(r_mask_flat_.unsqueeze(2), p_mask_expand_.unsqueeze(1))
        
        # (batch_size*num_r, rw_length， hidden_dim * 4)                                                              
        r_matching_states, _ = self._get_r_matching_representations(r_hiddens_, p_matching_expand_,
                                                                 similarity_matrix, mask_matrix)

        predict_sort_ = self.match_lstm(r_matching_states[sort_idx,:,:], 
                                  z_rw.view(z_rw.size(0)*z_rw.size(1), -1)[sort_idx,:],
                                  r_mask_flat_[sort_idx,:])
        predict = predict_sort_[revert_idx,:].contiguous().view(r_mask.size(0), r_mask.size(1)) # (batch, num_r)
        
        return predict
        
class DifferentiateMatchLSTMRankingModel(BasicNLPModel):
    def __init__(self, words_lookup, args):
        super(DifferentiateMatchLSTMRankingModel, self).__init__(words_lookup, args)
        
        self.num_classes = args.num_classes
        self.hidden_dim = args.hidden_dim
        self.is_cuda = args.cuda
        self.model_type = args.model_type

        # create an encoder layer
        self.match_lstm = DifferentiateMatchLSTMEncoder(args)
        
        # create a loss function 
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        
        
    def _create_embed_layer(self, vocab_size, emb_dim):
        embed_layer = nn.Embedding(vocab_size, emb_dim)
        embed_layer.weight.requires_grad = True
        return embed_layer
        
        
    def forward(self, q, p, r, q_mask, p_mask, r_mask):
        
        q_embeddings = self.embedding_layer(q) #(batch_size, q_length, embedding_dim)
        p_embeddings = self.embedding_layer(p) #(batch_size, p_length, embedding_dim)
        rw_embeddings = self.embedding_layer(r) #(batch_size, num_r, r_length, embedding_dim)         
        
        z_q = torch.ones_like(q).float().cuda() # (batch, q_length)
        z_p = torch.ones_like(p).float().cuda() # (batch, p_length)
        z_r = torch.ones_like(r).float().cuda() # (batch, num_r, r_length)
        
        predict = self.match_lstm(q_embeddings, p_embeddings, rw_embeddings, 
                                  z_q, z_p, z_r, q_mask, p_mask, r_mask)
        
        return predict
    
    def loss(self, predict, label):
        prediction_loss = self.loss_func(predict, label) # (batch_size, )
        supervised_loss = torch.mean(prediction_loss)
                
        return supervised_loss


# In[ ]:


class CoMatchLSTMEncoder(RelationalMatchLSTMEncoder):
    def __init__(self, args):
        super(CoMatchLSTMEncoder, self).__init__(args)
        # create an diff layer
        
#         self.linear_proj_rq = nn.Sequential(
#                 nn.Linear(args.hidden_dim*4, args.hidden_dim),
#                 nn.ReLU()
#             )
#         self.linear_proj_rp = nn.Sequential(
#                 nn.Linear(args.hidden_dim*4, args.hidden_dim),
#                 nn.ReLU()
#             )
        
        match_lstm_args = copy.deepcopy(args)
        match_lstm_args.model_type = 'RNN'
        match_lstm_args.layer_num = 1
        match_lstm_args.embedding_dim = match_lstm_args.hidden_dim * 4
        match_lstm_args.hidden_dim = args.mlstm_hidden_dim
        match_lstm_args.num_classes = 1
        self.match_lstm = ClassificationEncoder(match_lstm_args)
        
    def forward(self, q_embeddings, p_embeddings, r_embeddings, z_q, z_p, z_rw, q_mask, p_mask, r_mask):
        # q encoding
        q_len = torch.sum(q_mask, dim=1).cpu().data.numpy()
        sort_idx = np.argsort(-q_len)
        idx_dict = {sort_idx[i_]: i_ for i_ in range(q_len.shape[0])}
        revert_idx = np.array([idx_dict[i_] for i_ in range(q_len.shape[0])])
        
        q_hiddens_sort_ = self.encoder(q_embeddings[sort_idx,:,:], z_q[sort_idx,:], q_mask[sort_idx,:]) #(batch_size, hidden_dim, q_length)
        q_hiddens = q_hiddens_sort_[revert_idx, :, :].contiguous()
        
        # p encoding
        p_len = torch.sum(p_mask, dim=1).cpu().data.numpy()
        p_sort_idx = np.argsort(-p_len)
        p_idx_dict = {p_sort_idx[i_]: i_ for i_ in range(p_len.shape[0])}
        p_revert_idx = np.array([p_idx_dict[i_] for i_ in range(p_len.shape[0])])
        
        p_hiddens_sort_ = self.encoder(p_embeddings[p_sort_idx,:,:], z_p[p_sort_idx,:], p_mask[p_sort_idx,:])
        p_hiddens = p_hiddens_sort_[p_revert_idx, :, :].contiguous()

        # r encoding
        r_mask_flat_ = r_mask.view(r_mask.size(0)*r_mask.size(1), -1) #(batch_size*num_r, rw_length)
        r_embeddings_flat_ = r_embeddings.view(r_mask.size(0)*r_mask.size(1), r_mask.size(2), -1)
        
        r_len = torch.sum(r_mask_flat_, dim=1).cpu().data.numpy()
        sort_idx = np.argsort(-r_len)
        idx_dict = {sort_idx[i_]: i_ for i_ in range(r_len.shape[0])}
        revert_idx = np.array([idx_dict[i_] for i_ in range(r_len.shape[0])])
        
        r_hiddens_sort_ = self.encoder(r_embeddings_flat_[sort_idx,:,:], 
                                   z_rw.view(z_rw.size(0)*z_rw.size(1), -1)[sort_idx,:],
                                   r_mask_flat_[sort_idx,:]) #(batch*num_r, hidden_dim, rw_length)
        
        r_hiddens_ = r_hiddens_sort_[revert_idx,:,:].contiguous()
        
        # R-Q attention
        q_hiddens_expand_ = q_hiddens.unsqueeze(1).expand(q_mask.size(0), r_mask.size(1), -1, q_mask.size(1))
        q_hiddens_expand_ = q_hiddens_expand_.contiguous().view(q_mask.size(0)*r_mask.size(1), -1, q_mask.size(1))
        q_mask_expand_ = q_mask.unsqueeze(1).expand(q_mask.size(0), r_mask.size(1), q_mask.size(1))
        q_mask_expand_ = q_mask_expand_.contiguous().view(-1, q_mask.size(1))
        
        rq_similarity_matrix = self._calculate_similarity_matrix(r_hiddens_, q_hiddens_expand_)
        rq_mask_matrix = torch.bmm(r_mask_flat_.unsqueeze(2), q_mask_expand_.unsqueeze(1))
        
        # (batch_size*num_r, rw_length， hidden_dim * 4)                                                              
        rq_matching_states, _ = self._get_r_matching_representations(r_hiddens_, q_hiddens_expand_,
                                                                 rq_similarity_matrix, rq_mask_matrix)
#         rq_matching_states = self.linear_proj_rq(rq_matching_states)
        
        # R-P attention
        # expand to (batch*num_r, hidden_dim, p_length) and (batch*num_r, p_length)
        p_hiddens_expand_ = p_hiddens.unsqueeze(1).expand(p_mask.size(0), r_mask.size(1), -1, p_mask.size(1))
        p_hiddens_expand_ = p_hiddens_expand_.contiguous().view(p_mask.size(0)*r_mask.size(1), -1, p_mask.size(1))
        p_mask_expand_ = p_mask.unsqueeze(1).expand(p_mask.size(0), r_mask.size(1), p_mask.size(1))
        p_mask_expand_ = p_mask_expand_.contiguous().view(-1, p_mask.size(1))

        rp_similarity_matrix = self._calculate_similarity_matrix(r_hiddens_, p_hiddens_expand_)
        rp_mask_matrix = torch.bmm(r_mask_flat_.unsqueeze(2), p_mask_expand_.unsqueeze(1))
        
        # (batch_size*num_r, p_length, hidden_dim * 4)
        rp_matching_states, _ = self._get_r_matching_representations(r_hiddens_, p_hiddens_expand_,
                                                                 rp_similarity_matrix, rp_mask_matrix)
#         rp_matching_states = self.linear_proj_rp(rp_matching_states)

#         r_matching_states = torch.cat([rq_matching_states, rp_matching_states], dim=2)
        r_matching_states = rp_matching_states
        predict_sort_ = self.match_lstm(r_matching_states[sort_idx,:,:], 
                                  z_rw.view(z_rw.size(0)*z_rw.size(1), -1)[sort_idx,:],
                                  r_mask_flat_[sort_idx,:])
        predict = predict_sort_[revert_idx,:].contiguous().view(r_mask.size(0), r_mask.size(1)) # (batch, num_r)
        
        return predict
        
class CoMatchLSTMRankingModel(BasicNLPModel):
    def __init__(self, words_lookup, args):
        super(CoMatchLSTMRankingModel, self).__init__(words_lookup, args)
        
        self.num_classes = args.num_classes
        self.hidden_dim = args.hidden_dim
        self.is_cuda = args.cuda
        self.model_type = args.model_type

        # create an encoder layer
        self.match_lstm = CoMatchLSTMEncoder(args)
        
        # create a loss function 
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        
        
    def _create_embed_layer(self, vocab_size, emb_dim):
        embed_layer = nn.Embedding(vocab_size, emb_dim)
        embed_layer.weight.requires_grad = True
        return embed_layer
        
        
    def forward(self, q, p, r, q_mask, p_mask, r_mask):
        
        q_embeddings = self.embedding_layer(q) #(batch_size, q_length, embedding_dim)
        p_embeddings = self.embedding_layer(p) #(batch_size, p_length, embedding_dim)
        rw_embeddings = self.embedding_layer(r) #(batch_size, num_r, r_length, embedding_dim)         
        
        z_q = torch.ones_like(q).float().cuda() # (batch, q_length)
        z_p = torch.ones_like(p).float().cuda() # (batch, p_length)
        z_r = torch.ones_like(r).float().cuda() # (batch, num_r, r_length)
        
        predict = self.match_lstm(q_embeddings, p_embeddings, rw_embeddings, 
                                  z_q, z_p, z_r, q_mask, p_mask, r_mask)
        
        return predict
    
    def loss(self, predict, label):
        prediction_loss = self.loss_func(predict, label) # (batch_size, )
        supervised_loss = torch.mean(prediction_loss)
                
        return supervised_loss

