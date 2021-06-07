#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dataset import TextDataset
import numpy as np
import sys, os, json
import gzip
from colored import fg, attr, bg

from env import JerichoEnv
from tqdm import tqdm
from jericho import *

from pair_dataset import TemplateActionParser
from pair_dataset_bert import BERTStateState2ActionDataset, calc_score

import codecs

import random


# In[ ]:


import jericho
print(jericho.__version__)


# In[ ]:



from transformers import BertTokenizer

import transformers
# print(transformers.__version__)


# In[ ]:


def _bert_tokenize_original_sas_data(pretrain_path, games, data_dir):
    
    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    
    def _tokenize_observation(instance):
        
#         doc = nlp_pipe(line.strip())
        
        new_inst = json.loads(instance)
        info = new_inst['observations']['obs'].split(' | ')
        for i in range(3):
            spacy_tokens = info[i].split(' ')
#             print(info[i])
#             print(spacy_tokens)
            text_tokens = []
            for spacy_token in spacy_tokens:
                if spacy_token == '':
                    text_tokens.append(spacy_token)
                else:
                    tokens = tokenizer.tokenize(spacy_token)
                    for token in tokens:
                        text_tokens.append(token)
#             tokens = tokenizer.tokenize(info[i])
#             text_tokens = [token for token in tokens if token != ' ']
            info[i] = ' '.join(text_tokens)
        new_inst['observations']['obs'] = ' | '.join(info)
        
        for idx, action_group in enumerate(new_inst['valid_actions']):
            action_tuple = action_group[0]
            info = action_tuple['observations'].split(' | ')
            for i in range(3):
                spacy_tokens = info[i].split(' ')
                text_tokens = []
                for spacy_token in spacy_tokens:
                    if spacy_token == '':
                        text_tokens.append(spacy_token)
                    else:
                        tokens = tokenizer.tokenize(spacy_token)
                        for token in tokens:
                            text_tokens.append(token)
#                 tokens = tokenizer.tokenize(info[i])
#                 text_tokens = [token for token in tokens if token != ' ']
                info[i] = ' '.join(text_tokens)
            new_inst['valid_actions'][idx][0]['observations'] = ' | '.join(info)
        
        return new_inst
    
    for game_name in games:
        print('# LOADING game data {} ...'.format(game_name))

        f = open(os.path.join(data_dir, '{}.sas.wt_traj.with_rouge.tok'.format(game_name)), "r")
        instances = f.readlines()
        
        fout = open(os.path.join(data_dir, '{}.sas.wt_traj.with_rouge.bert_tok2'.format(game_name)), "w")

        for idx, instance in enumerate(instances):
            instance_tok = _tokenize_observation(instance)
            fout.write(json.dumps(instance_tok) + '\n')
            
        f.close()
        fout.close()
            
# pretrain_path = '/dccstor/gaot1/MultiHopReason/comprehension_tasks/narrativeqa/passage_ranker/bert-base-uncased/'

# train_games = ['905', 'acorncourt', 'advent', 'adventureland', 'afflicted', 'awaken', 
#                'balances', 'deephome', 'dragon', 'enchanter', 'inhumane', 'library', 
#                'moonlit', 'omniquest', 'pentari', 'reverb', 'snacktime', 'sorcerer', 'zork1']
# dev_games = ['zork3', 'detective', 'ztuu', 'jewel', 'zork2']
# test_games = ['temple', 'gold', 'karn', 'zenon', 'wishbringer']

# games = train_games + dev_games + test_games
# # games = ['905']

# data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/jecc_sup/"

# _bert_tokenize_original_sas_data(pretrain_path, games, data_dir)
            


# In[ ]:





# In[ ]:


class BERTForwardPredictionSet(object):
    '''
    '''
    def __init__(self):
        self.pairs = []
        self.num_positive = 0
        self.num_tuples = 0
        
        self.CLS_TOKEN = 101
        self.SEP_TOKEN = 102
        self.SPLIT_TOKEN = 5
        
        self.action2tuples = {}
        
    def add_one(self, state, next_states, actions, wt_next_state, wt_action):
        pair_id = len(self.pairs)
        self.pairs.append({'state':state, 'next_states':next_states, 'actions':actions, 
                           'wt_next_state':wt_next_state, 'wt_action':wt_action})
        
        if wt_action not in self.action2tuples:
            self.action2tuples[wt_action] = [(pair_id, -1)]
        else:
            self.action2tuples[wt_action].append((pair_id, -1))
        
        for idx, (next_state, action) in enumerate(zip(next_states, actions)):
            if action not in self.action2tuples:
                self.action2tuples[action] = [(pair_id, idx)]
            else:
                self.action2tuples[action].append((pair_id, idx))
        
        self.num_positive += 1
        self.num_tuples += len(actions)
        
    def get_pairs(self):
        return self.pairs
    
    def size(self):
        return len(self.pairs)
    
    def get_eval_concat_samples(self, batch_idx, num_negative=9, truncate_num=0):
        concat_inputs = []
        outputs = []
        labels = []
        positives = []
        candidates = []
        
        max_in_len = -1
        max_out_len = -1
        
        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            
            state = pair_dict_['state']
            wt_action = pair_dict_['wt_action']
            wt_action_str = pair_dict_['wt_action_str']
            wt_next_state = pair_dict_['wt_next_state']
            
            wt_question = [self.CLS_TOKEN] + state + [self.SEP_TOKEN] + wt_action
            if truncate_num > 0:
                wt_question = wt_question[:truncate_num]
            if len(wt_question) > max_in_len:
                max_in_len = len(wt_question)
            concat_inputs.append(wt_question)
            
            wt_next_state = [self.CLS_TOKEN] + wt_next_state + [self.SEP_TOKEN]
            if truncate_num > 0:
                wt_next_state = wt_next_state[:truncate_num]
            if len(wt_next_state) > max_out_len:
                max_out_len = len(wt_next_state)
            outputs.append(wt_next_state)
            
            # sample other actions under the state                
            for neg_idx in range(len(pair_dict_["actions"])):
                action = pair_dict_['actions'][neg_idx]
                next_state = pair_dict_['next_states'][neg_idx]
                
                question = [self.CLS_TOKEN] + state + [self.SEP_TOKEN] + action
                if truncate_num > 0:
                    question = question[:truncate_num]
                if len(question) > max_in_len:
                    max_in_len = len(question)
                concat_inputs.append(question)

                next_state = [self.CLS_TOKEN] + next_state + [self.SEP_TOKEN]
                if truncate_num > 0:
                    next_state = next_state[:truncate_num]
                if len(next_state) > max_out_len:
                    max_out_len = len(next_state)
                outputs.append(next_state)
            
        return concat_inputs, outputs, max_in_len, max_out_len
    
    def get_eval_triple_concat_samples(self, batch_idx, num_negative=9, truncate_num=0):
        concat_triples_list = []
        labels = []
        
        max_lens = []
        
        state_list = []
        action_list = []
        next_state_list = []
        
        for i, idx in enumerate(batch_idx):
            max_len = -1
            concat_triples = []
            
            pair_dict_ = self.pairs[idx]
            
            state = pair_dict_['state']
            wt_action = pair_dict_['wt_action']
            wt_action_str = pair_dict_['wt_action_str']
            wt_next_state = pair_dict_['wt_next_state']
            
            state_list.append(state)
            action_list.append(wt_action)
            next_state_list.append(wt_next_state)
            
            # sample other actions under the state
#             neg_sample_idxs = random.choices(list(range(len(pair_dict_["actions"]))), k=num_negative)
#             for neg_idx in neg_sample_idxs:
            for neg_idx in range(len(pair_dict_["actions"])):
                action = pair_dict_['actions'][neg_idx]
                next_state = pair_dict_['next_states'][neg_idx]
                
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
        
#         print('num_negative:', num_negative)
#         print('list length', len(state_list))
        for input_id in range(len(state_list)):
            max_len = -1
            concat_triples = []
            input_seq = action_list[input_id] + [self.SPLIT_TOKEN] + state_list[input_id]
#             input_seq = action_list[input_id]
            for output_id in range(len(next_state_list)):
                seq = [self.CLS_TOKEN] + next_state_list[output_id] + [self.SEP_TOKEN]
                seq += input_seq

                if truncate_num > 0:
                    seq = seq[:truncate_num]
                if len(seq) > max_len:
                    max_len = len(seq)
                concat_triples.append(seq)
                
            max_lens.append(max_len)
            concat_triples_list.append(concat_triples)
            
        return concat_triples_list, max_lens
    
    
    def get_eval_triple_samples(self, batch_idx, num_negative=9, truncate_num=0):
        states = []
        actions = []
        outputs = []
        labels = []
        
        max_in_len = -1
        max_a_len = -1
        max_out_len = -1
        
        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            
            state = pair_dict_['state']
            wt_action = pair_dict_['wt_action']
            wt_action_str = pair_dict_['wt_action_str']
            wt_next_state = pair_dict_['wt_next_state']
            
            wt_question = [self.CLS_TOKEN] + state + [self.SEP_TOKEN]
            if truncate_num > 0:
                wt_question = wt_question[:truncate_num]
            if len(wt_question) > max_in_len:
                max_in_len = len(wt_question)
            states.append(wt_question)
            
            wt_action = [self.CLS_TOKEN] + wt_action + [self.SEP_TOKEN]
            if len(wt_action) > max_a_len:
                max_a_len = len(wt_action)
            actions.append(wt_action)
            
            wt_next_state = [self.CLS_TOKEN] + wt_next_state + [self.SEP_TOKEN]
            if truncate_num > 0:
                wt_next_state = wt_next_state[:truncate_num]
            if len(wt_next_state) > max_out_len:
                max_out_len = len(wt_next_state)
            outputs.append(wt_next_state)
            
            # sample other actions under the state                
            for neg_idx in range(len(pair_dict_["actions"])):
                action = pair_dict_['actions'][neg_idx]
                next_state = pair_dict_['next_states'][neg_idx]
                
                question = [self.CLS_TOKEN] + state + [self.SEP_TOKEN]
                if truncate_num > 0:
                    question = question[:truncate_num]
                if len(question) > max_in_len:
                    max_in_len = len(question)
                states.append(question)
                
                action = [self.CLS_TOKEN] + action + [self.SEP_TOKEN]
                if len(action) > max_a_len:
                    max_a_len = len(action)
                actions.append(action)

                next_state = [self.CLS_TOKEN] + next_state + [self.SEP_TOKEN]
                if truncate_num > 0:
                    next_state = next_state[:truncate_num]
                if len(next_state) > max_out_len:
                    max_out_len = len(next_state)
                outputs.append(next_state)
            
        return states, actions, outputs, max_in_len, max_a_len, max_out_len
    
    def get_eval_comatch_concat_samples(self, batch_idx, num_negative=9, truncate_num=0):
        seq1_list = []
        seq2_list = []
        labels = []
        
        max_len1_list = []
        max_len2_list = []
        
        state_list = []
        action_list = []
        next_state_list = []
        
        for i, idx in enumerate(batch_idx):
            max_len = -1
            concat_triples = []
            
            pair_dict_ = self.pairs[idx]
            
            state = pair_dict_['state']
            wt_action = pair_dict_['wt_action']
            wt_action_str = pair_dict_['wt_action_str']
            wt_next_state = pair_dict_['wt_next_state']
            
            state_list.append(state)
            action_list.append(wt_action)
            next_state_list.append(wt_next_state)
            
            for neg_idx in range(len(pair_dict_["actions"])):
                action = pair_dict_['actions'][neg_idx]
                next_state = pair_dict_['next_states'][neg_idx]
                
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
        
        for input_id in range(len(state_list)):
            max_len1 = -1
            max_len2 = -1
            concat_pairs1 = []
            concat_pairs2 = []
#             input_seq = action_list[input_id]
            for output_id in range(len(next_state_list)):
                seq1 = [self.CLS_TOKEN] + next_state_list[output_id] + [self.SEP_TOKEN] + state_list[input_id]
                seq2 = [self.CLS_TOKEN] + next_state_list[output_id] + [self.SEP_TOKEN] + action_list[input_id]

                if truncate_num > 0:
                    seq1 = seq1[:truncate_num]
                if len(seq1) > max_len1:
                    max_len1 = len(seq1)
                concat_pairs1.append(seq1)
                
                if truncate_num > 0:
                    seq2 = seq2[:truncate_num]
                if len(seq2) > max_len2:
                    max_len2 = len(seq2)
                concat_pairs2.append(seq2)
                
            max_len1_list.append(max_len1)
            seq1_list.append(concat_pairs1)
            max_len2_list.append(max_len2)
            seq2_list.append(concat_pairs2)
            
        return seq1_list, seq2_list, max_len1_list, max_len2_list
    
    def check_eval_triples(self, vocab):
#         concat_triples_list = []
#         labels = []
        
#         max_lens = []
        
#         state_list = []
#         action_list = []
#         next_state_list = []

        total_cand = 0
        total_state = 0
        max_cand = 0
        
        for idx in range(len(self.pairs)):
#             max_len = -1
            concat_triples = []
            
            pair_dict_ = self.pairs[idx]
            
            state = pair_dict_['state']
            wt_action = pair_dict_['wt_action']
            wt_action_str = pair_dict_['wt_action_str']
            wt_next_state = pair_dict_['wt_next_state']
            
#             state_list.append(state)
#             action_list.append(wt_action)
#             next_state_list.append(wt_next_state)
            
#             # sample other actions under the state
#             neg_sample_idxs = random.choices(list(range(len(pair_dict_["actions"]))), k=num_negative)
#             for idx in neg_sample_idxs:
#                 action = pair_dict_['actions'][idx]
#                 next_state = pair_dict_['next_states'][idx]
                
#                 state_list.append(state)
#                 action_list.append(action)
#                 next_state_list.append(next_state)

            num_cand = 0
            obs_dict = {}
            for neg_idx in range(len(pair_dict_["actions"])):
                action = pair_dict_['actions'][neg_idx]
                next_state = pair_dict_['next_states'][neg_idx]

                next_state_words = [vocab[wid] for wid in next_state]
                next_state_str = ' '.join(next_state_words)
                
#                 if len(pair_dict_["actions"]) == 27:
#                     action_words = [vocab[wid] for wid in action]
#                     print(' '.join(action_words))
#                     print(next_state_str)
                
                if next_state_str not in obs_dict:
                    obs_dict[next_state_str] = 1
                    num_cand += 1
                    
            if num_cand > max_cand:
                max_cand = num_cand
            total_cand += num_cand
#             if len(pair_dict_["actions"]) > max_cand:
#                 max_cand = len(pair_dict_["actions"])
#             total_cand += len(pair_dict_["actions"])
            total_state += 1
        print('max number of cand:', max_cand)
        print('avg number of cand: {}/{}={}'.format(total_cand, total_state, total_cand/total_state))
            
    def print_info(self):
        print('Number of walkthrough tuples: {}'.format(self.num_positive))
        print('Number of tuples: {}'.format(self.num_tuples))
        print('Number of unique actions: {}'.format(len(self.action2tuples)))


# In[ ]:


class BERTStateAction2StateDataset(BERTStateState2ActionDataset):
    
    def __init__(self, pretrain_path, data_dir, rom_dir, game2rom, 
                 train_games=None, dev_games=None, test_games=None, setting='same_games',
                 num_negative=20, truncate_num=384, freq_threshold=2):
        self.test_games = test_games
        super(BERTStateAction2StateDataset, self).__init__(pretrain_path, data_dir, rom_dir, game2rom,
                                                          train_games, dev_games, setting,
                                                          num_negative, truncate_num, freq_threshold)
        
    def load_dataset(self):
        
        self.data_sets = {}
        
        if self.setting == 'same_games':
            self.data_sets = self._load_pair_data_and_split(self.train_games)
        
        elif self.setting == 'transfer':
            # load train
            self.data_sets = self._load_pair_data_transfer(self.train_games, self.dev_games)
        
        # build vocab
        self._build_vocab()
        
    def _process_instance(self, instance):
        new_inst = json.loads(instance)
        info = new_inst['observations'].split('|')
        new_inst['observations'] = {'obs':' | '.join(info[:3]), 'action':info[3]}
        
        for idx, action_group in enumerate(new_inst['valid_actions']):
            action_tuple = action_group[0]
            info = action_tuple['observations'].split('|')
#             print(new_inst['valid_actions'][idx][0]['observations'])
            new_inst['valid_actions'][idx][0]['observations'] = ' | '.join(info)
#             print(new_inst['valid_actions'][idx][0]['observations'])
            
#         print(new_inst['observations']['obs'])
#         print(new_inst['observations']['action'])
        return new_inst
        
    def _load_pair_data_and_split(self, games, neg_removal=True):
        """
        Splitting trajectories with 8:1:1
        """
        datasets = {}
        datasets['train'] = BERTForwardPredictionSet()
        datasets['dev'] = BERTForwardPredictionSet()
        datasets['test'] = BERTForwardPredictionSet()
        
        avg_cand_rouge_l = 0
        total_cand = 0
        
        avg_wt_rouge_l = 0
        total_wt_cand = 0
        
        test_act_dict = {}
        
        for game_name in games:
#             rom_path = "../roms/jericho-game-suite/{}.z5".format(game_name)
            print('# LOADING game data {} ...'.format(game_name))
    
            num_unmatched_wt_action = 0
            
#             f = open(os.path.join(self.data_dir, '{}.sas.wt_traj.with_rouge.bert_tok'.format(game_name)), "r")
            f = open(os.path.join(self.data_dir, '{}.sas.wt_traj.with_rouge.tok'.format(game_name)), "r")
            instances = f.readlines()
            
#             instances = [self._process_instance(instance.lower()) for instance in instances]
            instances = [json.loads(instance) for instance in instances]

            for idx, instance in enumerate(instances):
                if idx == len(instances) - 1:
                    continue
            
                state = instance['observations']['obs']
                wt_next_state = instances[idx + 1]['observations']['obs']
                wt_action_origin = instances[idx + 1]['observations']['action']
                
                next_states = []
                actions = []
                
#                 print(instance)
#                 break
                
                wt_match_flag = False
                for valid_act_group in instance['valid_actions']:
                    valid_act_tuple = valid_act_group[0]
                    action = valid_act_tuple['a']
                    next_state = valid_act_tuple['observations']
                    
                    next_states.append(next_state)
                    actions.append(action)
                    
                    if 'rougel' in valid_act_tuple:
                        rouge_scores.append(valid_act_tuple['rougel'])
                    else:
                        rouge_scores.append('NA')
                    
                    if next_state == wt_next_state:
                        wt_action = action
                        wt_match_flag = True
                        
                if not wt_match_flag:
#                     print('unmatched action: \'{}\''.format(wt_action_origin))
                    wt_action = wt_action_origin
                    num_unmatched_wt_action += 1
        
                rouge_score = calc_score([wt_next_state], [state])
                avg_wt_rouge_l += rouge_score
                total_wt_cand += 1

                actions__ = []
                next_states__ = []

                rouge_scores__ = []

                obs_dict = {wt_next_state:1}
                for neg_idx in range(len(actions)):
                    action = actions[neg_idx]
                    next_state = next_states[neg_idx]

                    if action.startswith('drop') and len(action.split(' ')) == 2:
                        continue

                    if next_state not in obs_dict:
#                         rouge_score = calc_score([next_state], [state])                            
                        rouge_score = rouge_scores[neg_idx]

                        obs_dict[next_state] = 1
                        actions__.append(action)
                        next_states__.append(next_state)
                        rouge_scores__.append((len(rouge_scores__), rouge_score))

                sorted_rouge_scores = sorted(rouge_scores__, key = lambda x:x[1])
                actions_ = []
                next_states_ = []
                for (neg_idx, _) in sorted_rouge_scores:
                    action = actions__[neg_idx]
                    next_state = next_states__[neg_idx]

                    actions_.append(action)
                    next_states_.append(next_state)

                    avg_cand_rouge_l += rouge_score
                    total_cand += 1

                    if len(actions_) == 15:
                        break

                if game_name not in dev_games and game_name not in test_games:
                    datasets['train'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                else:                                                    
                    if game_name in dev_games:
                        datasets['dev'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                    else:
                        datasets['test'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
            print('# unmatched actions in the game: {}'.format(num_unmatched_wt_action))
            
        for k, data_set in datasets.items():
            print('# {} set'.format(k))
            data_set.print_info()
            
        print('# averaged rouge-L between walkthrough dev/test (s, s\'): {}'.format(avg_wt_rouge_l/total_wt_cand))
        print('# averaged rouge-L between dev/test (s, s\'): {}'.format(avg_cand_rouge_l/total_cand))

        return datasets
    
    def _load_pair_data_transfer(self, games, dev_games, neg_removal=True):
        """
        Splitting dev trajectories with 5:5
        """
        datasets = {}
        datasets['train'] = BERTForwardPredictionSet()
        datasets['dev'] = BERTForwardPredictionSet()
        datasets['test'] = BERTForwardPredictionSet()
        
        if self.test_games is None:
            test_games = []
        else:
            test_games = self.test_games
        
        avg_cand_rouge_l = 0
        total_cand = 0
        
        avg_wt_rouge_l = 0
        total_wt_cand = 0
        
        dev_num_drop = 0
        dev_num_pick = 0
        dev_num_burn = 0
        test_num_drop = 0
        test_num_pick = 0
        test_num_burn = 0

        dev_act_dict = {}
        test_act_dict = {}
        
        def _preprocess_action(action):
            action = action.lower()

            if action == 'n':
                action = 'north'
            elif action == 's':
                action = 'south'
            elif action == 'e':
                action = 'east'
            elif action == 'w':
                action = 'west'
            elif action == 'se':
                action = 'southeast'
            elif action == 'sw':
                action = 'southwest'
            elif action == 'ne':
                action = 'northeast'
            elif action == 'nw':
                action = 'northwest'
            elif action == 'u':
                action = 'up'
            elif action == 'd':
                action = 'down'
            return action
        
        for game_name in games + dev_games + test_games:
#             rom_path = "../roms/jericho-game-suite/{}.z5".format(game_name)
            print('# LOADING game data {} ...'.format(game_name))
    
            num_unmatched_wt_action = 0
            
            f = open(os.path.join(self.data_dir, '{}.sas.wt_traj.with_rouge.bert_tok2'.format(game_name)), "r")
#             f = open(os.path.join(self.data_dir, '{}.sas.wt_traj.with_rouge.tok'.format(game_name)), "r")
            instances = f.readlines()
            
#             instances = [self._process_instance(instance.lower()) for instance in instances]
            instances = [json.loads(instance) for instance in instances]

            for idx, instance in enumerate(instances):
                if idx == len(instances) - 1:
                    continue
            
#                 info = instance['observations']['obs'].split(' | ')
#                 state = ' | '.join([info[0], info[2]])
                state = instance['observations']['obs']
                wt_next_state = instances[idx + 1]['observations']['obs']
                wt_action_origin = instances[idx + 1]['observations']['action']
            
                wt_action_origin = _preprocess_action(wt_action_origin)
                
                next_states = []
                actions = []
                rouge_scores = []
                
                wt_match_flag = False
                for valid_act_group in instance['valid_actions']:
                    valid_act_tuple = valid_act_group[0]
                    action = valid_act_tuple['a']
                    next_state = valid_act_tuple['observations']
                    
                    next_states.append(next_state)
                    actions.append(action)
                    
                    if 'rougel' in valid_act_tuple:
                        rouge_scores.append(valid_act_tuple['rougel'])
                    else:
                        rouge_scores.append('NA')
                    
#                     print(next_state)
#                     print(wt_next_state)
                    
                    if next_state == wt_next_state:
#                         wt_action = action
                        wt_action = wt_action_origin
                        wt_match_flag = True
                        
                if not wt_match_flag:
#                     print('unmatched action: \'{}\''.format(wt_action_origin))
                    wt_action = wt_action_origin
                    num_unmatched_wt_action += 1
        
                rouge_score = calc_score([wt_next_state], [state])
                avg_wt_rouge_l += rouge_score
                total_wt_cand += 1

                actions__ = []
                next_states__ = []

                rouge_scores__ = []

                obs_dict = {wt_next_state:1}
                for neg_idx in range(len(actions)):
                    action = actions[neg_idx]
                    next_state = next_states[neg_idx]

                    if action.startswith('drop') and len(action.split(' ')) == 2:
                        continue

                    if next_state not in obs_dict:
#                         rouge_score = calc_score([next_state], [state])                            
                        rouge_score = rouge_scores[neg_idx]

                        obs_dict[next_state] = 1
                        actions__.append(action)
                        next_states__.append(next_state)
                        rouge_scores__.append((len(rouge_scores__), rouge_score))

                sorted_rouge_scores = sorted(rouge_scores__, key = lambda x:x[1])
                actions_ = []
                next_states_ = []
                for (neg_idx, _) in sorted_rouge_scores:
                    action = actions__[neg_idx]
                    next_state = next_states__[neg_idx]

                    actions_.append(action)
                    next_states_.append(next_state)

                    avg_cand_rouge_l += rouge_score
                    total_cand += 1

                    if len(actions_) == 15:
                        break

                if game_name not in dev_games and game_name not in test_games:
                    datasets['train'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                else:                                                    
                    if game_name in dev_games:
                        datasets['dev'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                    else:
                        datasets['test'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
            print('# unmatched actions in the game: {}'.format(num_unmatched_wt_action))
            
        for k, data_set in datasets.items():
            print('# {} set'.format(k))
            data_set.print_info()
        
        print('# averaged rouge-L between walkthrough dev/test (s, s\'): {}'.format(avg_wt_rouge_l/total_wt_cand))
        print('# averaged rouge-L between dev/test (s, s\'): {}'.format(avg_cand_rouge_l/total_cand))
        
#         for act_dict in [dev_act_dict, test_act_dict]:
#             num_key = 0
#             num_bias = 0
#             num_unbias = 0
#             num_dup = 0
#             for k, v in act_dict.items():
#                 num_key += 1
#                 num_bias += len(v)
#                 if len(v) == 1:
#                     num_unbias += 1
#                     for k2, v2 in v.items():
#                         if v2 > 1 and len(act_dict) == len(test_act_dict):
#                             print(k + '\t' + k2, '\t', v2)
# #                             print('count:', v2)
#                         num_dup += v2
#                 else:
#                     if len(act_dict) == len(test_act_dict):
#                         print(k, '\t', len(v))
#                         for k2, v2 in v.items():
#                             print(k2)
#             print('num action: {}, num bias: {}, avg bias: {}'.format(num_key, num_bias, num_bias/num_key))
#             print('num unbias action: {}, num dup: {}, avg dup: {}'.format(num_unbias, num_dup, num_dup/num_unbias))
            
#         print('dev num dropped:', dev_num_drop)
#         print('dev num picked:', dev_num_pick)
#         print('dev num burn:', dev_num_burn)
        
#         print('test num dropped:', test_num_drop)
#         print('test num picked:', test_num_pick)
#         print('test num burn:', test_num_burn)

        return datasets
    
    def _numeralize_pairs(self, pairs):
        '''
        numeralize passages in training pair lists
        '''
        ret_pair_list = []
        for pair_dict_ in pairs:
            new_pair_dict_ = {}

            for k, v in pair_dict_.items():
                if k == 'state' or k == 'wt_next_state':
                    new_pair_dict_[k] = self.tokenizer.convert_tokens_to_ids(v.split(' '))
                elif k == 'wt_action':
                    new_pair_dict_[k] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(v))
                    new_pair_dict_['wt_action_str'] = v
                elif k == 'next_states':
                    new_pair_dict_[k] = []
                    for seq in v:
                        new_pair_dict_[k].append(self.tokenizer.convert_tokens_to_ids(seq.split(' ')))
                elif k == 'actions':
                    new_pair_dict_[k] = []
                    for seq in v:
                        new_pair_dict_[k].append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq)))
            ret_pair_list.append(new_pair_dict_)
        return ret_pair_list
    
        
    def get_batch_concat(self, set_id, batch_idx, num_negative=-1):
        """
        randomly select a batch from a dataset
        Inputs:
            batch_idx: 
        Outputs (all numpy arrays are sorted according to q_length):
            x_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            a_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            y_vec -- numpy array of binary labels, numpy array in shape of (batch_size,)
            x_mask -- numpy array of masks
        """
        
        if num_negative < 0:
            num_negative = self.num_negative
        
        data_set = self.data_sets[set_id]
        concat_inputs, outputs, max_in_len, max_out_len = data_set.get_eval_concat_samples(batch_idx, 
                                                                               num_negative=num_negative, 
                                                                               truncate_num=self.truncate_num)

        i_masks_ = []
        o_masks_ = []

        for i, q in enumerate(concat_inputs):
            concat_inputs[i] = q + (max_in_len - len(q)) * [0]
            i_masks_.append([1] * len(q) + [0] * (max_in_len - len(q)))
            
        for i, a in enumerate(outputs):
            outputs[i] = a + (max_out_len - len(a)) * [0]
            o_masks_.append([1] * len(a) + [0] * (max_out_len - len(a)))
            
        i_mat = np.array(concat_inputs, dtype=np.int64)
        i_mask = np.array(i_masks_, dtype=np.int64)
        o_mat = np.array(outputs, dtype=np.int64)
        o_mask = np.array(o_masks_, dtype=np.int64)
        y_vec = np.array(range(len(concat_inputs)), dtype=np.int64)
        
        return i_mat, o_mat, y_vec, i_mask, o_mask
    
    def get_eval_batch_concat(self, set_id, batch_idx, num_negative=-1):
        
        if num_negative < 0:
            num_negative = self.num_negative
        
        data_set = self.data_sets[set_id]
        concat_inputs, outputs, max_in_len, max_out_len = data_set.get_eval_concat_samples(batch_idx, 
                                                                               num_negative=num_negative, 
                                                                               truncate_num=self.truncate_num)

        i_masks_ = []
        o_masks_ = []

        for i, q in enumerate(concat_inputs):
            concat_inputs[i] = q + (max_in_len - len(q)) * [0]
            i_masks_.append([1] * len(q) + [0] * (max_in_len - len(q)))
            
        for i, a in enumerate(outputs):
            outputs[i] = a + (max_out_len - len(a)) * [0]
            o_masks_.append([1] * len(a) + [0] * (max_out_len - len(a)))
            
        i_mat = np.array(concat_inputs, dtype=np.int64)
        i_mask = np.array(i_masks_, dtype=np.int64)
        o_mat = np.array(outputs, dtype=np.int64)
        o_mask = np.array(o_masks_, dtype=np.int64)
        y_vec = np.array(range(len(concat_inputs)), dtype=np.int64)
        
        return i_mat, o_mat, y_vec, i_mask, o_mask
    
    def get_batch_triple_concat(self, set_id, batch_idx, num_negative=-1):        
        if num_negative < 0:
            num_negative = self.num_negative
        
        data_set = self.data_sets[set_id]
        seq1_list, seq, max_lens = data_set.get_triple_concat_samples_from_one_list(batch_idx, 
                                                                            num_negative=num_negative, 
                                                                            truncate_num=self.truncate_num)

        masks_list = []
        mat_list = []
        
        for (seqs, max_len) in zip(seqs_list, max_lens):
            i_masks_ = []
            for i, seq in enumerate(seqs):
                seqs[i] = seq + (max_len - len(seq)) * [0]
                i_masks_.append([1] * len(seq) + [0] * (max_len - len(seq)))
            
            i_mat = np.array(seqs, dtype=np.int64)
            i_mask = np.array(i_masks_, dtype=np.int64)
            mat_list.append(i_mat)
            masks_list.append(i_mask)
            
        y_vec = np.array(range(len(seqs)), dtype=np.int64)
        
        return mat_list, y_vec, masks_list
    
    def get_eval_batch_triple_concat(self, set_id, batch_idx, num_negative=9):        
        if num_negative < 0:
            num_negative = self.num_negative
        
        data_set = self.data_sets[set_id]
        seqs_list, max_lens = data_set.get_eval_triple_concat_samples(batch_idx, 
                                                                            num_negative=num_negative, 
                                                                            truncate_num=self.truncate_num)

        masks_list = []
        mat_list = []
        
        for (seqs, max_len) in zip(seqs_list, max_lens):
            i_masks_ = []
            for i, seq in enumerate(seqs):
                seqs[i] = seq + (max_len - len(seq)) * [0]
                i_masks_.append([1] * len(seq) + [0] * (max_len - len(seq)))
            
            i_mat = np.array(seqs, dtype=np.int64)
            i_mask = np.array(i_masks_, dtype=np.int64)
            mat_list.append(i_mat)
            masks_list.append(i_mask)
            
        y_vec = np.array(range(len(seqs)), dtype=np.int64)
        
        return mat_list, y_vec, masks_list
    
    
    def get_eval_batch_triple(self, set_id, batch_idx, num_negative=-1):
        
        if num_negative < 0:
            num_negative = self.num_negative
        
        data_set = self.data_sets[set_id]
        states, actions, outputs, max_in_len, max_a_len, max_out_len = data_set.get_eval_triple_samples(
                                                                                batch_idx, 
                                                                               num_negative=num_negative, 
                                                                               truncate_num=self.truncate_num)

        i_masks_ = []
        a_masks_ = []
        o_masks_ = []

        for i, q in enumerate(states):
            states[i] = q + (max_in_len - len(q)) * [0]
            i_masks_.append([1] * len(q) + [0] * (max_in_len - len(q)))
            
        for i, a in enumerate(actions):
            actions[i] = a + (max_a_len - len(a)) * [0]
            a_masks_.append([1] * len(a) + [0] * (max_a_len - len(a)))
            
        for i, a in enumerate(outputs):
            outputs[i] = a + (max_out_len - len(a)) * [0]
            o_masks_.append([1] * len(a) + [0] * (max_out_len - len(a)))
            
        i_mat = np.array(states, dtype=np.int64)
        i_mask = np.array(i_masks_, dtype=np.int64)
        a_mat = np.array(actions, dtype=np.int64)
        a_mask = np.array(a_masks_, dtype=np.int64)
        o_mat = np.array(outputs, dtype=np.int64)
        o_mask = np.array(o_masks_, dtype=np.int64)
        y_vec = np.array(range(len(states)), dtype=np.int64)
        
        return i_mat, a_mat, o_mat, y_vec, i_mask, a_mask, o_mask
    
    def get_eval_batch_comatch_concat(self, set_id, batch_idx, num_negative=9):        
        if num_negative < 0:
            num_negative = self.num_negative
        
        data_set = self.data_sets[set_id]
        seq1_list, seq2_list, max_len1_list, max_len2_list = data_set.get_eval_comatch_concat_samples(batch_idx, 
                                                                            num_negative=num_negative, 
                                                                            truncate_num=self.truncate_num)
        
        def _get_tensors(seq_list, max_len_list):
            masks_list = []
            mat_list = []
            for (seqs, max_len) in zip(seq_list, max_len_list):
                i_masks_ = []
                for i, seq in enumerate(seqs):
                    seqs[i] = seq + (max_len - len(seq)) * [0]
                    i_masks_.append([1] * len(seq) + [0] * (max_len - len(seq)))

                i_mat = np.array(seqs, dtype=np.int64)
                i_mask = np.array(i_masks_, dtype=np.int64)
                mat_list.append(i_mat)
                masks_list.append(i_mask)
            return mat_list, masks_list
        
        mat1_list, masks1_list = _get_tensors(seq1_list, max_len1_list)
        mat2_list, masks2_list = _get_tensors(seq2_list, max_len2_list)
            
        y_vec = np.array(range(len(seq1_list)), dtype=np.int64)
        
        return mat1_list, mat2_list, y_vec, masks1_list, masks2_list
    


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

if __name__=='__main__':
    
    data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/jecc_sup/"
    
    train_games = ['905', 'acorncourt', 'advent', 'adventureland', 'afflicted', 'awaken', 
                   'balances', 'deephome', 'dragon', 'enchanter', 'inhumane', 'library', 
                   'moonlit', 'omniquest', 'pentari', 'reverb', 'snacktime', 'sorcerer', 'zork1']
    dev_games = ['zork3', 'detective', 'ztuu', 'jewel', 'zork2']
    test_games = ['temple', 'gold', 'karn', 'zenon', 'wishbringer']
    
    games = train_games + dev_games + test_games
#     train_games = ['905']
#     dev_games = ['acorncourt']
#     test_games = ['advent']
    
    rom_dir = '../roms/jericho-game-suite/'
    game2rom = find_game_roms(games, rom_dir)
    print(game2rom)
    
    pretrain_path = '/dccstor/gaot1/MultiHopReason/comprehension_tasks/narrativeqa/passage_ranker/bert-base-uncased/'
    
    game_task_data = BERTStateAction2StateDataset(pretrain_path, data_dir, rom_dir=rom_dir, game2rom=game2rom,
                                              train_games=train_games, dev_games=dev_games,
                                              test_games = test_games,
                                              setting='transfer')

    game_task_data.data_sets['train'].check_eval_triples(game_task_data.idx_2_word)
    game_task_data.data_sets['dev'].check_eval_triples(game_task_data.idx_2_word)
    game_task_data.data_sets['test'].check_eval_triples(game_task_data.idx_2_word)

    i_mat, a_mat, o_mat, y_vec, i_mask, a_mask, o_mask = game_task_data.get_eval_batch_triple('train', [2])

    for i in range(i_mat.shape[0]):
    #     display_sentences(game_task_data, states[i], input_sent_masks[i])
        game_task_data.display_sentence(i_mat[i])
        game_task_data.display_sentence(a_mat[i])
        game_task_data.display_sentence(o_mat[i])
    #     print(output_sent_masks[i])
    #     display_sentences(game_task_data, i_mat[i], is_masks[i])
    
    mat1_list, mat2_list, y_vec, masks1_list, masks2_list = game_task_data.get_eval_batch_comatch_concat('train', 
                                                                                                         [2])
    
    for i in range(mat1_list[0].shape[0]):
    #     display_sentences(game_task_data, states[i], input_sent_masks[i])
        game_task_data.display_sentence(mat1_list[0][i])
        game_task_data.display_sentence(mat2_list[0][i])
    


# In[ ]:


#     i_mat, a_mat, o_mat, y_vec, i_mask, a_mask, o_mask = game_task_data.get_eval_batch_triple('train', [2])

#     for i in range(i_mat.shape[0]):
#     #     display_sentences(game_task_data, states[i], input_sent_masks[i])
#         game_task_data.display_sentence(i_mat[i])
#         game_task_data.display_sentence(a_mat[i])
#         game_task_data.display_sentence(o_mat[i])


# In[ ]:


# print(len(states))

# for i in range(len(states)):
#     game_task_data.display_sentence(states[i])
#     game_task_data.display_sentence(actions[i])
#     game_task_data.display_sentence(outputs[i])
# #     print(concat_inputs[i])
# #     print(outputs[i])
#     print('')
    


# In[ ]:


# mat_list, y_vec, masks_list = game_task_data.get_eval_batch_triple_concat('dev', [rand_indices[0]])

# j=5
# for i in range(len(mat_list[j])):
#     print(len(mat_list[j][i]))
#     game_task_data.display_sentence(mat_list[j][i])

# print(y_vec)


# In[ ]:


# def display_observation(dataset, x, output_action=True, action_only=False):
#     """
#     Display a suquence of word index
#     Inputs:
#         x -- input sequence of word indices, (sequence_length,)
#     Outputs:
#         None
#     """
#     # apply threshold
#     segment = 0
#     if not action_only:
#         print('LOOK:')
#     for word_index in x:
#         word = dataset.idx_2_word[word_index]
#         if word == '|' or word == '[SEP]':
#             segment += 1
#             if segment == 1 and not action_only:
#                 print('\nINVENTORY:')
#             elif segment == 2 and not action_only:
#                 print('\nOBSERVATION:')
#             elif segment == 3 and output_action:
#                 sys.stdout.write('\nACTION:')
#         if word == '[PAD]' or word == '|' or word == '[SEP]' or word == '[CLS]':
#             continue
#         if segment < 3 and not action_only:
#             sys.stdout.write(" " + word)
#         elif segment == 3 and output_action:
#             sys.stdout.write(" " + word)
#     if not action_only:
#         sys.stdout.write("\n")
#     sys.stdout.flush()


# In[ ]:


#     i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_eval_batch_concat('train', [203])
#     print(y_vec)

#     for i in range(i_mat.shape[0]):
#         display_observation(game_task_data, i_mat[i])
#         display_observation(game_task_data, o_mat[i], output_action=False)
#     #     print(concat_inputs[i])
#     #     print(outputs[i])
#         print('')


# In[ ]:


# num_dev_instance = game_task_data.data_sets['dev'].size()
# rand_indices = list(range(num_dev_instance))
# random.shuffle(rand_indices)

# for idx in rand_indices:

#     i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_eval_batch_concat('dev', [idx])
#     if i_mat.shape[0] > 0: # == 16:
#         print(i_mat.shape)

#         display_observation(game_task_data, i_mat[0], output_action=False)

#         for i in range(i_mat.shape[0]):
#             display_observation(game_task_data, i_mat[i], action_only=True)
#         print('')
#         #     game_task_data.display_sentence(o_mat[i])

#         output_indices = list(range(o_mat.shape[0]))
#         random.shuffle(output_indices)
#         for output_idx in output_indices:
#             print(output_idx)
#             display_observation(game_task_data, o_mat[output_idx], output_action=False)

#         cmd = input()
        
# #         break


# In[ ]:


# print(np.expand_dims(y_vec, axis=1))
# # print(type(y_vec.))

# for seq, y, mask in zip(mat_list, np.expand_dims(y_vec, axis=1), masks_list):
#     print(type(seq))
#     print(type(y))
#     print(type(mask))


# In[ ]:


#     mat_list, y_vec, masks_list = game_task_data.get_eval_batch_triple_concat('dev', [0])

#     print(len(mat_list))
#     print(len(masks_list))
#     print(y_vec)

# #     print(mat_list[0])
# #     print(masks_list[0])

#     j=2
#     for i in range(len(mat_list[j])):
#         print(len(mat_list[j][i]))
#         game_task_data.display_sentence(mat_list[j][i])

# i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_eval_batch_concat('dev', [1])
# print(i_mat.shape)
# for i in range(i_mat.shape[0]):
#     game_task_data.display_sentence(i_mat[i])
#     game_task_data.display_sentence(o_mat[i])
#     print('')


# 

# # transfer setting
# 
#     max number of cand: 27
#     avg number of cand: 1846/136=13.573529411764707
#     max number of cand: 30
#     avg number of cand: 1952/135=14.459259259259259
#     
# ## removing actions with same observations
#     max number of cand: 27
#     avg number of cand: 1835/136=13.492647058823529
#     max number of cand: 25
#     avg number of cand: 1886/135=13.97037037037037
#     
#     
# # same game setting
#     
#     max number of cand: 57
#     avg number of cand: 4758/238=19.991596638655462
#     max number of cand: 48
#     avg number of cand: 3518/231=15.229437229437229
# 
# ## removing actions with same observations
#     
#     max number of cand: 45
#     avg number of cand: 3998/238=16.798319327731093
#     max number of cand: 46
#     avg number of cand: 3037/231=13.147186147186147

# In[ ]:


# seqs_list, max_lens = game_task_data.data_sets['train'].get_triple_concat_samples_from_one_list([0,1], 
#                                                                     num_negative=4)

# print(max_lens)

# j=3
# for i in range(len(seqs_list[j])):
#     game_task_data.display_sentence(seqs_list[j][i])
# #     print(seqs_list[j][i])


# In[ ]:


# def generating_evaluation_data(game_task_data, set_name, out_file_name):
#     total = game_task_data.data_sets[set_name].size()
    
#     i_mat_list = []
#     i_mask_list = []
#     o_mat_list = []
#     o_mask_list = []
    
#     for i in range(total // 2):
#         i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_batch_concat(set_name, 
#                                                                               [i*2, i*2+1], 
#                                                                               num_negative=4)
#         i_mat_list.append(i_mat)
#         i_mask_list.append(i_mask)
#         o_mat_list.append(o_mat)
#         o_mask_list.append(o_mask)
    
# # def loading_dev_and_testing_data():


# In[ ]:


#     i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_batch_concat('train', [4,5], num_negative=4)

#     print(y_vec)
#     print(i_mat.shape)


# In[ ]:


# set_id = 'train'
# data_set = game_task_data.data_sets[set_id]
# batch_idx = np.random.randint(0, data_set.size(), size=40)
# print(batch_idx)

# x_mat, a_mat, y_vec, x_mask, a_mask = game_task_data.get_train_batch(40, inst_format='concat')
# print(x_mat.shape)
# print(a_mat.shape)
# print(x_mask.shape)
# print(a_mask.shape)
# print(y_vec.shape)


# In[ ]:



# rom_path = "../roms/jericho-game-suite/zork1.z5" # "../roms/jericho-game-suite/zork1.z5"

# bindings = load_bindings(rom_path)

# print(bindings['grammar'])


# unmatched action: turn page
# unmatched action: drop all except torch and lamp
# unmatched action: get knife and bag
# cannot recognize verb: odysseus
# unmatched action: odysseus
# unmatched action: drop rusty knife
# cannot recognize verb: inflate
# unmatched action: inflate pile
# unmatched action: launch
# unmatched action: get out of boat
# unmatched action: dig sand
# unmatched action: dig sand
# unmatched action: dig sand
# unmatched action: dig sand
# unmatched action: get rusty knife
# unmatched action: get nasty knife
# unmatched action: kill thief with nasty knife
# unmatched action: kill thief with nasty knife
# unmatched action: kill thief with nasty knife
# unmatched action: attack thief with nasty knife
# unmatched action: drop rusty knife
# unmatched action: drop nasty knife
# cannot recognize verb: examine
# unmatched action: examine map

# In[ ]:




