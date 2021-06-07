#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque

from jecc_dataset_bert import BERTStateAction2StateDataset

from tqdm import tqdm

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # 3.15it/s

torch.manual_seed(9527)
np.random.seed(9527)
random.seed(9527)


# In[ ]:


import os
print(os.environ["CUDA_VISIBLE_DEVICES"])
print(torch.__version__)


# In[ ]:



class Argument():
    def __init__(self):
        self.hidden_dim = 200
        self.mlstm_hidden_dim = 100
        self.embedding_dim = 100
#         self.embedding_dim = 300
        self.num_classes = 2
        self.kernel_size = 3
        self.layer_num = 1
        self.fine_tuning = False
        self.cuda = True
        self.lambda_l2 = 0.05
        self.model_type = "LSTM"
        self.cell_type = "GRU"
        self.batch_size = 10
        self.input_topk = 32
        self.keep_prob = 0.8
        self.predict_target_topk = 5
        self.save_path = 'trained_models'
#         self.model_prefix = 'hotpot_reranker_model_h%d.with_anchor_with_el'%self.hidden_dim
        self.model_prefix = 'tmp_model'
        self.load_model = False
        self.load_path = '.'
        
args = Argument()
print(vars(args))


# In[ ]:


def find_game_roms(games, rom_dir):
    print('#number of games: {}'.format(len(games)))

    roms = os.listdir(rom_dir)
    game2rom = {}
    logs = []
    for game in games:
        for rom in roms:
            if rom.startswith(game + '.z'):
                game2rom[game] = rom
    #             print('find {} for {}'.format(rom, game))
                logs.append('find {} for {}'.format(rom, game))
        if game not in game2rom:
            print('cannot find rom for {}'.format(game))

    print('#number of roms founds: {}'.format(len(logs)))
    
    return game2rom

import importlib
import jecc_dataset_bert
importlib.reload(jecc_dataset_bert)
from jecc_dataset_bert import BERTStateAction2StateDataset

# data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/supervised/"
data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/jecc_sup/"

train_games = ['905', 'acorncourt', 'advent', 'adventureland', 'afflicted', 'awaken', 
               'balances', 'deephome', 'dragon', 'enchanter', 'inhumane', 'library', 
               'moonlit', 'omniquest', 'pentari', 'reverb', 'snacktime', 'sorcerer', 'zork1']
dev_games = ['zork3', 'detective', 'ztuu', 'jewel', 'zork2']
test_games = ['temple', 'gold', 'karn', 'zenon', 'wishbringer']

games = train_games + dev_games + test_games
    
rom_dir = '../roms/jericho-game-suite/'
game2rom = find_game_roms(games, rom_dir)
print(game2rom)

# games = ['zork1', 'zork3']

pretrain_path = '/dccstor/gaot1/MultiHopReason/comprehension_tasks/narrativeqa/passage_ranker/bert-base-uncased/'

# game_task_data = BERTStateAction2StateDataset(pretrain_path, data_dir, rom_dir=rom_dir, game2rom=game2rom,
#                                           train_games=games, dev_games=games,
#                                           setting='same_games', num_negative=4)

game_task_data = BERTStateAction2StateDataset(pretrain_path, data_dir, rom_dir=rom_dir, game2rom=game2rom,
                                              train_games=train_games, dev_games=dev_games, 
                                              test_games = test_games, truncate_num=512,
                                              setting='transfer', num_negative=4)

game_task_data.data_sets['train'].check_eval_triples(game_task_data.idx_2_word)
game_task_data.data_sets['dev'].check_eval_triples(game_task_data.idx_2_word)
game_task_data.data_sets['test'].check_eval_triples(game_task_data.idx_2_word)


# In[ ]:


from torch.nn.utils import rnn

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

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
#             lens = input_lengths.data.cpu().numpy()
            lens = input_lengths
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


from transformers import BertModel, BertTokenizer
import logging

class CoMatchBertForwardRanking(nn.Module):
    def __init__(self, args, max_length, num_class, pretrain_path, blank_padding=True):
        super().__init__()
        
        args.hidden = args.hidden_dim
        
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768

        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        
        self.diff_linear = nn.Sequential(
                nn.Linear(self.hidden_size*4, args.hidden),
                nn.ReLU()
            )

        self.match_lstm = EncoderRNN(args.hidden*2, int(args.mlstm_hidden_dim/2), 1, False, True, 1-args.keep_prob, False)
#         self.output_layer = nn.Linear(args.mlstm_hidden_dim, args.num_classes)
        
        self.num_classes = num_class
        
        self.siamese_output_layer = nn.Linear(args.hidden*2, self.num_classes)
        self.output_layer = nn.Linear(args.mlstm_hidden_dim, self.num_classes)
        self.loss = nn.CrossEntropyLoss()
        
    def bert_vars(self):
        """
        Return the variables of the generator.
        """
        params = list(self.bert.parameters())

        return params
    
    def comatch_vars(self):
        """
        Return the variables of the generator.
        """
        params = list(self.diff_linear.parameters()) + list(
            self.match_lstm.parameters()) + list(
            self.output_layer.parameters())

        return params
        
        
    def _get_matching_representations(self, q_hiddens, p_hiddens, q_mask, p_mask):
        '''
        inputs -- (B, L1, H), (B, L2, H)
        '''
        similarity_matrix = torch.bmm(q_hiddens, p_hiddens.transpose(1, 2)) # (B, L1, L2)
        mask_matrix = torch.bmm(q_mask.unsqueeze(2), p_mask.unsqueeze(1)) #(B, L1, L2)
        
        neg_inf = -1.0e6
        attention_softmax = F.softmax(similarity_matrix + (1 * mask_matrix) * neg_inf, dim=2)
        
        # shape: (B, L1, H)                                                            
        q_hiddens_tilda = torch.bmm(attention_softmax, p_hiddens)

        # shape: (B, L1, 4*H)
        q_matching_states = torch.cat([q_hiddens, q_hiddens_tilda,
                                      q_hiddens - q_hiddens_tilda,
                                      q_hiddens * q_hiddens_tilda], dim=2)
        return q_matching_states
        
    def forward_siamese(self, i_hiddens, a_hiddens, o_hiddens, i_mask, a_mask, o_mask):
        '''
        inputs -- all (B, H)
        '''
        i_hiddens_expand_ = i_hiddens.unsqueeze(1).expand(i_mask.size(0), o_mask.size(0), -1) # (B, B, H)
        a_hiddens_expand_ = a_hiddens.unsqueeze(1).expand(i_mask.size(0), o_mask.size(0), -1) # (B, B, H)
        o_hiddens_expand_ = o_hiddens.unsqueeze(0).expand(i_mask.size(0), o_mask.size(0), -1) # (B, B, H)
        i_hiddens_expand_ = i_hiddens_expand_.contiguous().view(i_mask.size(0)*o_mask.size(0), -1) # (BB, H)
        a_hiddens_expand_ = a_hiddens_expand_.contiguous().view(i_mask.size(0)*o_mask.size(0), -1) # (BB, H)
        o_hiddens_expand_ = o_hiddens_expand_.contiguous().view(i_mask.size(0)*o_mask.size(0), -1) # (BB, H)

        diff_oi = torch.cat([o_hiddens_expand_, i_hiddens_expand_,
                             o_hiddens_expand_ - i_hiddens_expand_,
                             o_hiddens_expand_ * i_hiddens_expand_], dim=1) #(BB, 4H)
        diff_oi = self.diff_linear(diff_oi) #(BB, H)
        
        diff_oa = torch.cat([o_hiddens_expand_, a_hiddens_expand_,
                             o_hiddens_expand_ - a_hiddens_expand_,
                             o_hiddens_expand_ * a_hiddens_expand_], dim=1) #(BB, 4H)
        diff_oa = self.diff_linear(diff_oa) #(BB, H)
        
        co_match_inputs = torch.cat([diff_oi, diff_oa], dim=1) #(BB, 2H)
        
        predict = self.siamese_output_layer(co_match_inputs).view(i_mask.size(0), -1) #(B, B)
        
        return predict
    
        
    def forward(self, i, a, o, i_mask, a_mask, o_mask):
        """
        Args:
            i: (B, L), index of tokens
            a: (B, La), index of action tokens
            o: (B, L'), index of tokens
        Return:
            return (B, B) scores
        """
        i_hiddens, i_hiddens_cls = self.bert(i, attention_mask=i_mask) # (B, L1, H)
        a_hiddens, a_hiddens_cls = self.bert(a, attention_mask=a_mask) # (B, La, H)
        o_hiddens, o_hiddens_cls = self.bert(o, attention_mask=o_mask) # (B, L2, H)
        
        i_hiddens_expand_ = i_hiddens.unsqueeze(1).expand(i_mask.size(0), o_mask.size(0), 
                                                          i_hiddens.size(1), -1) # (B, B, L1, H)
        a_hiddens_expand_ = a_hiddens.unsqueeze(1).expand(i_mask.size(0), o_mask.size(0), 
                                                          a_hiddens.size(1), -1) # (B, B, La, H)
        o_hiddens_expand_ = o_hiddens.unsqueeze(0).expand(i_mask.size(0), o_mask.size(0), 
                                                          o_hiddens.size(1), -1) # (B, B, L2, H)
        i_hiddens_expand_ = i_hiddens_expand_.contiguous().view(i_mask.size(0)*o_mask.size(0), 
                                                                i_hiddens.size(1), -1) # (BB, H)
        a_hiddens_expand_ = a_hiddens_expand_.contiguous().view(i_mask.size(0)*o_mask.size(0), 
                                                                a_hiddens.size(1), -1) # (BB, H)
        o_hiddens_expand_ = o_hiddens_expand_.contiguous().view(i_mask.size(0)*o_mask.size(0), 
                                                                o_hiddens.size(1), -1) # (BB, H)
        
        i_mask_expand_ = torch.ones(i_mask.size(0) * o_mask.size(0), i_mask.size(1)).cuda() #(B1*B2, L1)
        a_mask_expand_ = torch.ones(i_mask.size(0) * o_mask.size(0), a_mask.size(1)).cuda() #(B1*B2, La)
        o_mask_expand_ = torch.ones(i_mask.size(0) * o_mask.size(0), o_mask.size(1)).cuda() #(B1*B2, L2)
        
        
        diff_oi = self._get_matching_representations(o_hiddens_expand_, i_hiddens_expand_, 
                                                     o_mask_expand_, i_mask_expand_) # (B1*B2, L2, 4*H)
        diff_oa = self._get_matching_representations(o_hiddens_expand_, a_hiddens_expand_, 
                                                     o_mask_expand_, a_mask_expand_) # (B1*B2, L2, 4*H)
        
        diff_oi = self.diff_linear(diff_oi) # (B1*B2, L2, H)
        diff_oa = self.diff_linear(diff_oa) # (B1*B2, L2, H)
        
        co_match_inputs = torch.cat([diff_oi, diff_oa], dim=2)
        o_len = torch.sum(o_mask_expand_, dim=1).cpu().data.numpy()
        
        co_match_hiddens = self.match_lstm(co_match_inputs, o_len) # (B1*B2, L2, H)

        neg_inf = -1.0e6
        co_match_hiddens = co_match_hiddens + (1 - o_mask_expand_.unsqueeze(2)) * neg_inf 
        
        max_co_match_hiddens = torch.max(co_match_hiddens, dim=1)[0] #(B1*B2, H)
        predict = self.output_layer(max_co_match_hiddens).view(i_mask.size(0), o_mask.size(0)) # (B1, B2)
#         predict += self.forward_siamese(i_hiddens_cls, a_hiddens_cls, o_hiddens_cls, i_mask, a_mask, o_mask)
        
        return predict
    


# 
# # train set
# Number of positive actions: 8732
# Number of total action candidates: 697315
# # dev set
# Number of positive actions: 2970
# Number of total action candidates: 366306
# # test set
# Number of positive actions: 2700
# Number of total action candidates: 247488
# size of the raw vocabulary: 14545
# size of the final vocabulary: 14333

# In[ ]:


class Argument():
    def __init__(self):
        self.hidden_dim = 200
        self.mlstm_hidden_dim = 100
        self.embedding_dim = 100
#         self.embedding_dim = 300
        self.num_classes = 1
        self.kernel_size = 3
        self.layer_num = 1
        self.fine_tuning = False
        self.cuda = True
        self.lambda_l2 = 0.05
        self.model_type = "LSTM"
        self.cell_type = "GRU"
        self.batch_size = 10
        self.input_topk = 32
        self.keep_prob = 0.8
        self.predict_target_topk = 5
        self.save_path = 'trained_models'
#         self.model_prefix = 'hotpot_reranker_model_h%d.with_anchor_with_el'%self.hidden_dim
        self.model_prefix = 'tmp_model'
        self.load_model = False
        self.load_path = '.'
        
args = Argument()
print(vars(args))

device = 'cuda'

ranker_model = CoMatchBertForwardRanking(args, max_length=384, num_class=1, pretrain_path=pretrain_path)
    
# load_model = True
# if load_model:
#     pre_trained = torch.load('comatch_bert_sas_jecc.2e-5.model.pt')
# #     pre_trained = torch.load('siamese_bert_sas.model.pt')
    
#     ranker_model.load_state_dict(pre_trained)

ranker_model.to(device)
print(ranker_model)

# bert_optimizer = torch.optim.Adam(ranker_model.bert_vars(), lr=2e-5)
# comatch_optimizer = torch.optim.Adam(ranker_model.comatch_vars(), lr=1e-3)

supervise_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, 
                                              ranker_model.parameters()), lr=2e-5)

# if args.cuda:
#     ranker_model.cuda()
    
sys.stdout.flush()


# In[ ]:


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print(get_n_params(ranker_model))


# In[ ]:


# train_losses = []
# train_accs = []
# dev_accs = [0.0]
# test_accs = [0.0]
# num_iteration = 20000
# display_iteration = 200
# test_iteration = 100
# queue_length = 400
# num_epoch = 40

# best_dev = 0.0
# best_dev_wt = 0.0
# batch_size_tuple = 2

# from tqdm.notebook import tqdm #_notebook

# if load_model:
#     with torch.no_grad():
#         ranker_model.eval()

#         for eval_set in ['dev', 'test']:
#             dev_correct = 0.0
#             dev_total = 0.0

#             dev_correct_wt = 0.0
#             dev_total_wt = 0.0

#             num_dev_instance = game_task_data.data_sets[eval_set].size()

#             for inst_id in tqdm(range(num_dev_instance)):
#                 i_mat, a_mat, o_mat, y_vec, i_mask, a_mask, o_mask = game_task_data.get_eval_batch_triple(eval_set, 
#                                                                                            [inst_id])

#                 i_mat_ = Variable(torch.from_numpy(i_mat)).to(device)
#                 a_mat_ = Variable(torch.from_numpy(a_mat)).to(device)
#                 o_mat_ = Variable(torch.from_numpy(o_mat)).to(device)
#                 i_mask_ = Variable(torch.from_numpy(i_mask)).float().to(device)
#                 a_mask_ = Variable(torch.from_numpy(a_mask)).float().to(device)
#                 o_mask_ = Variable(torch.from_numpy(o_mask)).float().to(device)
#                 y_vec_ = Variable(torch.from_numpy(y_vec)).to(device)
                
#                 shuffle_idx = list(range(o_mask_.size(0)))
#                 random.shuffle(shuffle_idx)
#                 shuffle_idx = np.array(shuffle_idx)
#                 o_mat_ = o_mat_[shuffle_idx,:]
#                 o_mask_ = o_mask_[shuffle_idx,:]

#                 predict = ranker_model(i_mat_, a_mat_, o_mat_, i_mask_, a_mask_, o_mask_)

#                 _, y_pred = torch.max(predict, dim=1)
#                 y_pred = shuffle_idx[y_pred.cpu().numpy()]

#                 dev_correct += (y_pred == y_vec).sum()
#                 dev_total += y_vec_.size(0)

# #                         y_pred = y_pred.cpu().data

#                 if y_vec_[0].item() == y_pred[0].item():
#                     dev_correct_wt += 1
# #                         if y_vec_[5].item() == y_pred[5].item():
# #                             dev_correct_wt += 1
#                 dev_total_wt += 1

#             if eval_set == 'dev':
#                 dev_accs.append(dev_correct / dev_total)
#                 if dev_correct / dev_total > best_dev:
#                     best_dev = dev_correct / dev_total
                    
#                 if dev_correct_wt/dev_total_wt > best_dev_wt:
#                     best_dev_wt = dev_correct_wt/dev_total_wt

#                 print('total: %d'%(dev_total))
#                 print('total wt: %d'%(dev_total_wt))

#                 print('dev acc: %f, best dev acc: %f' %(dev_correct/dev_total, best_dev))
#                 print('wt dev acc: %f, best wt dev acc: %f' %(dev_correct_wt/dev_total_wt, best_dev_wt))

#             else:
#                 test_accs.append(dev_correct / dev_total)
#                 print('total: %d'%(dev_total))
#                 print('total wt: %d'%(dev_total_wt))

#                 print('test acc: %f' %(dev_correct/dev_total))
#                 print('wt test acc: %f' %(dev_correct_wt/dev_total_wt))
#             sys.stdout.flush()

#         ranker_model.train()


# In[ ]:


train_losses = []
train_accs = []
dev_accs = [0.0]
test_accs = [0.0]
num_iteration = 20000
display_iteration = 200
test_iteration = 100
queue_length = 400
num_epoch = 40

best_dev = 0.0
best_dev_wt = 0.0
batch_size_tuple = 2

from tqdm.notebook import tqdm #_notebook

# i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_batch_concat('train', 
#                                                         list(range(tid * batch_size_tuple, 
#                                                         tid * batch_size_tuple + batch_size_tuple)), 
#                                                         num_negative=4)

# supervise_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, 
#                                               ranker_model.parameters()), lr=2e-4)

# i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_eval_batch_concat('dev', 
#                                                                            [inst_id])

for eid in range(num_epoch):
#for i in xrange(num_iteration):
    ranker_model.train()
    i = 0
    
    total = game_task_data.data_sets['train'].size()
    print('total num of tuples:', total)
    
    num_train_instance = game_task_data.data_sets['train'].size()
    tid_list = list(range(num_train_instance))
    random.shuffle(tid_list)

    for tid in tqdm(tid_list):
        i_mat, a_mat, o_mat, y_vec, i_mask, a_mask, o_mask = game_task_data.get_eval_batch_triple('train', [tid])
    
#     for tid in tqdm(range(total // batch_size_tuple)):
#         i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_batch_concat('train', 
#                                                                 list(range(tid * batch_size_tuple, 
#                                                                 tid * batch_size_tuple + batch_size_tuple)), 
#                                                                 num_negative=4)
    
        supervise_optimizer.zero_grad()
#         bert_optimizer.zero_grad()
#         comatch_optimizer.zero_grad()
        i_mat_ = Variable(torch.from_numpy(i_mat)).to(device)
        a_mat_ = Variable(torch.from_numpy(a_mat)).to(device)
        o_mat_ = Variable(torch.from_numpy(o_mat)).to(device)
        i_mask_ = Variable(torch.from_numpy(i_mask)).float().to(device)
        a_mask_ = Variable(torch.from_numpy(a_mask)).float().to(device)
        o_mask_ = Variable(torch.from_numpy(o_mask)).float().to(device)
        y_vec_ = Variable(torch.from_numpy(y_vec)).to(device)

        predict = ranker_model(i_mat_, a_mat_, o_mat_, i_mask_, a_mask_, o_mask_)
        
        supervised_loss = ranker_model.loss(predict, y_vec_)

        _, y_pred = torch.max(predict, dim=1)
        acc = np.float((y_pred == y_vec_).sum().cpu().data.item()) / y_vec_.size(0) # / args.batch_size
        train_accs.append(acc)
    
        supervised_loss.backward()
        supervise_optimizer.step()
#         bert_optimizer.step()
#         comatch_optimizer.step()
    
        i += 1
    
        if i % display_iteration == 0:
            print('train acc: %f supervised_loss: %f'%(np.mean(train_accs), 
                                           supervised_loss.cpu().data.item()))
            sys.stdout.flush()
    print('Epoch%d:'%(eid))
    game_task_data.display_sentence(i_mat[0])
    game_task_data.display_sentence(a_mat[0])
    game_task_data.display_sentence(o_mat[0])

    print('train acc: %f supervised_loss: %f'%(np.mean(train_accs), 
                                               supervised_loss.cpu().data.item()))
    train_losses = []
    train_accs = []
    
#     continue

    with torch.no_grad():
        ranker_model.eval()

        for eval_set in ['dev', 'test']:
        #         print('Training mode:', ranking_model.training)
            dev_correct = 0.0
            dev_total = 0.0

            dev_correct_wt = 0.0
            dev_total_wt = 0.0

#                     topk_coverage_dict = {1:0,2:0,5:0,10:0}
#                     topks = [1, 2, 5, 10]

            num_dev_instance = game_task_data.data_sets[eval_set].size()

#                     for inst_id in tqdm(range(num_dev_instance // batch_size_tuple)):
#                         i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_batch_concat('dev', 
#                                                                 list(range(inst_id * batch_size_tuple, 
#                                                                 inst_id * batch_size_tuple + batch_size_tuple)), 
#                                                                 num_negative=4)

            for inst_id in tqdm(range(num_dev_instance)):
                i_mat, a_mat, o_mat, y_vec, i_mask, a_mask, o_mask = game_task_data.get_eval_batch_triple(eval_set, 
                                                                                           [inst_id])

                i_mat_ = Variable(torch.from_numpy(i_mat)).to(device)
                a_mat_ = Variable(torch.from_numpy(a_mat)).to(device)
                o_mat_ = Variable(torch.from_numpy(o_mat)).to(device)
                i_mask_ = Variable(torch.from_numpy(i_mask)).float().to(device)
                a_mask_ = Variable(torch.from_numpy(a_mask)).float().to(device)
                o_mask_ = Variable(torch.from_numpy(o_mask)).float().to(device)
                y_vec_ = Variable(torch.from_numpy(y_vec)).to(device)

                shuffle_idx = list(range(o_mask_.size(0)))
                random.shuffle(shuffle_idx)
                shuffle_idx = np.array(shuffle_idx)
                o_mat_ = o_mat_[shuffle_idx,:]
                o_mask_ = o_mask_[shuffle_idx,:]
                
                predict = ranker_model(i_mat_, a_mat_, o_mat_, i_mask_, a_mask_, o_mask_)

                _, y_pred = torch.max(predict, dim=1)
                y_pred = shuffle_idx[y_pred.cpu().numpy()]

                dev_correct += (y_pred == y_vec).sum()
#                 dev_correct += np.float((y_pred == y_vec_).sum().cpu().data.item())
                dev_total += y_vec_.size(0)

#                         y_pred = y_pred.cpu().data

                if y_vec_[0].item() == y_pred[0].item():
                    dev_correct_wt += 1
#                         if y_vec_[5].item() == y_pred[5].item():
#                             dev_correct_wt += 1
                dev_total_wt += 1

            if eval_set == 'dev':
                dev_accs.append(dev_correct / dev_total)
                if dev_correct / dev_total > best_dev:
                    best_dev = dev_correct / dev_total
#                     print('new best dev:', best_dev, 'model saved at', 'comatch_siamese_bert_sas_jecc.model.pt')
#                     torch.save(ranker_model.state_dict(), 'comatch_siamese_bert_sas_jecc.model.pt')
                if dev_correct_wt/dev_total_wt > best_dev_wt:
                    best_dev_wt = dev_correct_wt/dev_total_wt
#                             print('new best dev:', best_dev, 'model saved at', 'siamese_bert_sas.model.rand50.pt')
#                             torch.save(ranker_model.state_dict(), 'siamese_bert_p1only.model.rand50.pt')
                    print('new best dev:', best_dev, 'model saved at', 'comatch_bert_with_res_jecc_no_wt_change.model.pt')
                    torch.save(ranker_model.state_dict(), 'comatch_bert_with_res_jecc_no_wt_change.model.pt')
#                     print('new best dev:', best_dev, 'model saved at', 'comatch_bert_sas_jecc.test_shuffled.model.pt')
#                     torch.save(ranker_model.state_dict(), 'comatch_bert_sas_jecc.test_shuffled.model.pt')

                print('total: %d'%(dev_total))
                print('total wt: %d'%(dev_total_wt))

                print('dev acc: %f, best dev acc: %f' %(dev_correct/dev_total, best_dev))
                print('wt dev acc: %f, best wt dev acc: %f' %(dev_correct_wt/dev_total_wt, best_dev_wt))

            else:
                test_accs.append(dev_correct / dev_total)
                print('total: %d'%(dev_total))
                print('total wt: %d'%(dev_total_wt))

                print('test acc: %f' %(dev_correct/dev_total))
                print('wt test acc: %f' %(dev_correct_wt/dev_total_wt))
            sys.stdout.flush()

        ranker_model.train()


# In[ ]:


# print(y_vec)
# print(predict[0])


# # for i in range(i_mat.shape[0]):
# #     game_task_data.display_sentence(i_mat[i])
# # print('')

# # for i in range(i_mat.shape[0]):
# #     game_task_data.display_sentence(o_mat[i])
# # print('')


# In[ ]:




