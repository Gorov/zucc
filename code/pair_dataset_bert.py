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


def _bert_tokenize_original_data(pretrain_path, games, data_dir):
    
    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    
    def _tokenize_observation(instance):
        
#         doc = nlp_pipe(line.strip())
        
        new_inst = json.loads(instance)
        info = new_inst['observations'].split('|')
        for i in range(3):
            tokens = tokenizer.tokenize(info[i])
            text_tokens = [token for token in tokens if token != ' ']
            info[i] = ' '.join(text_tokens)
        new_inst['observations'] = '|'.join(info)
        
        return new_inst
    
    for game_name in games:
        print('# LOADING game data {} ...'.format(game_name))

        f = open(os.path.join(data_dir, '{}.ssa.wt_traj.txt'.format(game_name)), "r")
        instances = f.readlines()
        
        fout = open(os.path.join(data_dir, '{}.ssa.wt_traj.bert_tok'.format(game_name)), "w")

        for idx, instance in enumerate(instances):
            instance_tok = _tokenize_observation(instance)
            fout.write(json.dumps(instance_tok) + '\n')
            
        f.close()
        fout.close()
            
# pretrain_path = '/dccstor/gaot1/MultiHopReason/comprehension_tasks/narrativeqa/passage_ranker/bert-base-uncased/'
# # tokenizer = BertTokenizer.from_pretrained(pretrain_path)
# # paragraph = 'forest path this is a path winding through a dimly lit forest. the path heads north south here. one particularly large tree with some low branches stands at the edge of the path.'
# # tokens = tokenizer.tokenize(paragraph)
# # print(tokens)
# data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/supervised/"

# games = ['905', 'acorncourt', 'advent', 'adventureland', 'afflicted', 'anchor', 'awaken', 
#          'balances', 'deephome', 'detective', 'dragon', 'enchanter', 'gold', 'inhumane', 
#          'jewel', 'karn', 'library', 'ludicorp', 'moonlit', 'omniquest', 'pentari', 'reverb', 
#          'snacktime', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'tryst205', 'yomomma', 
#          'zenon', 'zork1', 'zork3', 'ztuu']

# _bert_tokenize_original_data(pretrain_path, games, data_dir)
            


# In[ ]:


def _bert_tokenize_original_sas_data(pretrain_path, games, data_dir):
    
    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    
#     def _tokenize_observation(instance):
        
# #         doc = nlp_pipe(line.strip())
        
#         new_inst = json.loads(instance)
#         info = new_inst['observations'].split('|')
#         for i in range(3):
#             tokens = tokenizer.tokenize(info[i])
#             text_tokens = [token for token in tokens if token != ' ']
#             info[i] = ' '.join(text_tokens)
#         new_inst['observations'] = '|'.join(info)
        
#         for idx, action_group in enumerate(new_inst['valid_actions']):
#             action_tuple = action_group[0]
#             info = action_tuple['observations'].split('|')
#             for i in range(3):
#                 tokens = tokenizer.tokenize(info[i])
#                 text_tokens = [token for token in tokens if token != ' ']
#                 info[i] = ' '.join(text_tokens)
#             new_inst['valid_actions'][idx][0]['observations'] = '|'.join(info)
        
#         return new_inst
    
    def _tokenize_observation(instance):
        new_inst = json.loads(instance)
        info = new_inst['observations'].split('|')
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
            info[i] = ' '.join(text_tokens)
            
        new_inst['observations'] = '|'.join(info)
        
        for idx, action_group in enumerate(new_inst['valid_actions']):
            action_tuple = action_group[0]
            info = action_tuple['observations'].split('|')
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
                info[i] = ' '.join(text_tokens)
                
            new_inst['valid_actions'][idx][0]['observations'] = '|'.join(info)
        
        return new_inst
    
    for game_name in games:
        print('# LOADING game data {} ...'.format(game_name))

        f = open(os.path.join(data_dir, '{}.sas.wt_traj.tok'.format(game_name)), "r")
        instances = f.readlines()
        
        fout = open(os.path.join(data_dir, '{}.sas.wt_traj.bert_tok_new'.format(game_name)), "w")

        for idx, instance in enumerate(instances):
            instance_tok = _tokenize_observation(instance)
            fout.write(json.dumps(instance_tok) + '\n')
            
        f.close()
        fout.close()
            
# pretrain_path = '/dccstor/gaot1/MultiHopReason/comprehension_tasks/narrativeqa/passage_ranker/bert-base-uncased/'

# data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/zork_universe_sup/"

# # games = ['905', 'acorncourt', 'advent', 'adventureland', 'afflicted', 'anchor', 'awaken', 
# #          'balances', 'deephome', 'detective', 'dragon', 'enchanter', 'gold', 'inhumane', 
# #          'jewel', 'karn', 'library', 'ludicorp', 'moonlit', 'omniquest', 'pentari', 'reverb', 
# #          'snacktime', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'tryst205', 'yomomma', 
# #          'zenon', 'zork1', 'zork3', 'ztuu']

# games = ['zork1', 'zork3', 'enchanter', 'sorcerer']
# games = ['zork3', 'enchanter', 'sorcerer']
# # games = ['zork2', 'wishbringer']

# _bert_tokenize_original_sas_data(pretrain_path, games, data_dir)
            


# In[ ]:


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

rouge_beta = 1.0

def calc_score(candidate, refs):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
#     print(len(candidate))
#     print(len(refs))
    assert(len(candidate)==1)
    assert(len(refs)>0)         
    
#     print(refs)
    prec = []
    rec = []

    # split into tokens
    token_c = candidate[0].split(" ")

    for reference in refs:
        # split into tokens
        token_r = reference.split(" ")
        # compute the longest common subsequence
        lcs = my_lcs(token_r, token_c)
        prec.append(lcs/float(len(token_c)))
        rec.append(lcs/float(len(token_r)))

    prec_max = max(prec)
    rec_max = max(rec)
    
#     print('n:', len(token_c))
#     print('m:', len(token_r))
#     print('lcs:', lcs)
#     print('p:', prec_max)
#     print('r:', rec_max)

    if(prec_max!=0 and rec_max !=0):
        score = ((1 + rouge_beta**2)*prec_max*rec_max)/float(rec_max + rouge_beta**2*prec_max)
    else:
        score = 0.0
    return score


# In[ ]:


class BERTPair2SeqSet(object):
    '''
    '''
    def __init__(self):
        self.pairs = []
        self.num_positive = 0
        self.num_total = 0
        
        self.CLS_TOKEN = 101
        self.SEP_TOKEN = 102
        self.SPLIT_TOKEN = 5
        
    def add_one(self, input1, input2, positives, candidates):
        self.pairs.append({'input1':input1, 'input2':input2, 'positives':positives, 'candidates':candidates})
        self.num_positive += len(positives)
        self.num_total += len(candidates)
        
    def get_pairs(self):
        return self.pairs
    
    def size(self):
        return len(self.pairs)
    
    def get_samples_from_one_list(self, batch_idx, num_negative=10, truncate_num=0):
        x1 = []
        x2 = []
        positives = []
        candidates = []
        
        max_x1_len = -1
        max_x2_len = -1
        max_a_len = -1
        
        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = random.sample(pair_dict_['positives'], 1)[0]
            label = [self.CLS_TOKEN] + label + [self.SEP_TOKEN]
            if len(label) > max_a_len:
                max_a_len = len(label)
            positives.append(label)
            
            cand_list = []
#             num_neg_samples = min(len(pair_dict_["candidates"]), num_negative)
            neg_samples = random.choices(pair_dict_["candidates"], k=num_negative)
            for neg_sample in neg_samples:
                neg_sample = [self.CLS_TOKEN] + neg_sample + [self.SEP_TOKEN]
                if len(neg_sample) > max_a_len:
                    max_a_len = len(neg_sample)
                cand_list.append(neg_sample)
            candidates.append(cand_list)
            
            question = pair_dict_['input1']
            question = [self.CLS_TOKEN] + question + [self.SEP_TOKEN]
            if truncate_num > 0:
                question = question[:truncate_num]
            if len(question) > max_x1_len:
                max_x1_len = len(question)
                
            x1.append(question)

            passage = pair_dict_['input2']
            passage = [self.CLS_TOKEN] + passage + [self.SEP_TOKEN]
            if truncate_num > 0:
                passage = passage[:truncate_num]
            if len(passage) > max_x2_len:
                max_x2_len = len(passage)
                
            x2.append(passage)

        return x1, x2, positives, candidates, max_x1_len, max_x2_len, max_a_len
    
    def get_concat_samples_from_one_list(self, batch_idx, num_negative=10, truncate_num=0):
        concat_x = []
        positives = []
        candidates = []
        
        max_x_len = -1
        max_a_len = -1
        
        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = random.sample(pair_dict_['positives'], 1)[0]
            label = [self.CLS_TOKEN] + label + [self.SEP_TOKEN]
            if len(label) > max_a_len:
                max_a_len = len(label)
            positives.append(label)
            
            cand_list = []
#             num_neg_samples = min(len(pair_dict_["candidates"]), num_negative)
#             print('sampling {} actions'.format(num_negative))
            neg_samples = random.choices(pair_dict_["candidates"], k=num_negative)
#             print('{} actions sampled'.format(len(neg_samples)))
            for neg_sample in neg_samples:
                neg_sample = [self.CLS_TOKEN] + neg_sample + [self.SEP_TOKEN]
                if len(neg_sample) > max_a_len:
                    max_a_len = len(neg_sample)
                cand_list.append(neg_sample)
            candidates.append(cand_list)
#             print(len(cand_list))
            
#             question = pair_dict_['input1'] + [5] + pair_dict_['input2']
#             question = [self.CLS_TOKEN] + pair_dict_['input1'] + [self.SEP_TOKEN] + pair_dict_['input2']
            question = [self.CLS_TOKEN] + pair_dict_['input1'] + [self.SEP_TOKEN]
            if truncate_num > 0:
                question = question[:truncate_num]
            if len(question) > max_x_len:
                max_x_len = len(question)
                
            concat_x.append(question)
            
        return concat_x, positives, candidates, max_x_len, max_a_len
    
    def get_eval_samples_from_one_list(self, inst_idx, truncate_num=0):
        x1 = []
        x2 = []
        candidates = []
        
        max_x1_len = -1
        max_x2_len = -1
        max_a_len = -1
        
        pair_dict_ = self.pairs[inst_idx]
        cand_list = []
        y_list = []
        
        def _get_key_from_list(input_list):
            tmp_list = [str(x) for x in input_list]
            return ' '.join(tmp_list)
        
        positive_dict = {}
        for action in pair_dict_['positives']:
            action = [self.CLS_TOKEN] + action + [self.SEP_TOKEN]
            if len(action) > max_a_len:
                max_a_len = len(action)
            cand_list.append(action)
            y_list.append(1)
            positive_dict[_get_key_from_list(action)] = 1
            
        for action in pair_dict_["candidates"]:
            action = [self.CLS_TOKEN] + action + [self.SEP_TOKEN]
            key = _get_key_from_list(action)
            if key in positive_dict:
                continue
            
            if len(action) > max_a_len:
                max_a_len = len(action)
            cand_list.append(action)
            y_list.append(0)
            
        zip_list = list(zip(cand_list, y_list))
        random.shuffle(zip_list)
        cand_list = [x[0] for x in zip_list]
        y_list = [x[1] for x in zip_list]
            
        candidates.append(cand_list)

        question = pair_dict_['input1']
        question = [self.CLS_TOKEN] + question + [self.SEP_TOKEN]
        if truncate_num > 0:
            question = question[:truncate_num]
        if len(question) > max_x1_len:
            max_x1_len = len(question)

        x1.append(question)

        passage = pair_dict_['input2']
        passage = [self.CLS_TOKEN] + passage + [self.SEP_TOKEN]
        if truncate_num > 0:
            passage = passage[:truncate_num]
        if len(passage) > max_x2_len:
            max_x2_len = len(passage)

        x2.append(passage)
            
        return x1, x2, candidates, y_list, max_x1_len, max_x2_len, max_a_len
    
    
    def get_eval_concat_samples_from_one_list(self, inst_idx, truncate_num=0):
        concat_x = []
        positives = []
        candidates = []
        
        max_x_len = -1
        max_a_len = -1

        pair_dict_ = self.pairs[inst_idx]
        cand_list = []
        y_list = []
        
        def _get_key_from_list(input_list):
            tmp_list = [str(x) for x in input_list]
            return ' '.join(tmp_list)
        
        positive_dict = {}
        for action in pair_dict_['positives']:
            action = [self.CLS_TOKEN] + action + [self.SEP_TOKEN]
            if len(action) > max_a_len:
                max_a_len = len(action)
            cand_list.append(action)
            y_list.append(1)
            positive_dict[_get_key_from_list(action)] = 1
        
        for action in pair_dict_["candidates"]:
            action = [self.CLS_TOKEN] + action + [self.SEP_TOKEN]
            key = _get_key_from_list(action)
            if key in positive_dict:
                continue
            
            if len(action) > max_a_len:
                max_a_len = len(action)
            cand_list.append(action)
            y_list.append(0)
            
        zip_list = list(zip(cand_list, y_list))
        random.shuffle(zip_list)
        cand_list = [x[0] for x in zip_list]
        y_list = [x[1] for x in zip_list]
            
        candidates.append(cand_list)

#         question = [102] + pair_dict_['input1'] + [5] + pair_dict_['input2']
#         question = [self.CLS_TOKEN] + pair_dict_['input1'] + [self.SEP_TOKEN] + pair_dict_['input2']
        question = [self.CLS_TOKEN] + pair_dict_['input1'] + [self.SEP_TOKEN]

        if truncate_num > 0:
            question = question[:truncate_num]
        if len(question) > max_x_len:
            max_x_len = len(question)

        concat_x.append(question)
            
        return concat_x, candidates, y_list, max_x_len, max_a_len
            
    def print_info(self):
        print('Number of positive actions: {}'.format(self.num_positive))
        print('Number of total action candidates: {}'.format(self.num_total))


# In[ ]:


from pair_dataset import _preprocess_action, _match_action, _process_instance, _recover_root_template_action

class BERTStateState2ActionDataset(TextDataset):
    
    def __init__(self, pretrain_path, data_dir, rom_dir, game2rom, 
                 train_games=None, dev_games=None, setting='same_games',
                 num_negative=20, truncate_num=384, freq_threshold=2):        
        super(BERTStateState2ActionDataset, self).__init__(data_dir)
        self.pretrain_path = pretrain_path
        self.num_negative = num_negative
        self.truncate_num = truncate_num
        self.freq_threshold = freq_threshold
        
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_path)
        self.word_vocab, self.idx_2_word = self.load_bert_vocab()
        
#         self.word_vocab = {'<PAD>':0, '<START>':1, '<END>':2, '<UNK>':3, '<ANSWER>':4, '<SPLIT>':5, '|':6}
        
        self.rom_dir = rom_dir
        self.game2rom = game2rom
        self.setting = setting
        self.train_games = train_games
        self.dev_games = dev_games
        
        self.load_dataset()
        
    def load_bert_vocab(self):
        word_vocab = {}
        filein = codecs.open(os.path.join(self.pretrain_path, 'vocab.txt'), 'r', encoding='utf8')
        for line in filein:
            word = line.strip()
            word_vocab[word] = len(word_vocab)
        
        idx_2_word = {value: key for key, value in word_vocab.items()}
        return word_vocab, idx_2_word
        
    def load_dataset(self):
        
        self.data_sets = {}
        
        if self.setting == 'same_games':
            self.data_sets = self._load_pair_data_and_split(self.train_games)
        
        elif self.setting == 'transfer':
            # load train
            self.data_sets['train'] = self._load_pair_data(self.train_games)

            # load dev
            self.data_sets['dev'] = self._load_pair_data(self.dev_games)
    #         self.data_sets['test'] = self._load_pair_data(os.path.join(self.data_dir, 'test.tsv'))
        
        # build vocab
        self._build_vocab()
        
    def _load_pair_data_and_split(self, games, neg_removal=True):
        """
        Splitting trajectories with 8:1:1
        """
        datasets = {}
        datasets['train'] = BERTPair2SeqSet()
        datasets['dev'] = BERTPair2SeqSet()
        datasets['test'] = BERTPair2SeqSet()
        
        for game_name in games:
#             rom_path = "../roms/jericho-game-suite/{}.z5".format(game_name)
            print('# LOADING game data {} ...'.format(game_name))

            rom_path = os.path.join(self.rom_dir, self.game2rom[game_name])
            bindings = load_bindings(rom_path)
            act_par = TemplateActionParser(bindings)
            
            f = open(os.path.join(self.data_dir, '{}.ssa.wt_traj.tok'.format(game_name)), "r")
            instances = f.readlines()
            
            instances = [_process_instance(instance.lower()) for instance in instances]

            for idx, instance in enumerate(instances):
                if idx == len(instances) - 1:
                    continue
            
                input1 = instance['observations']['obs']
                
                input2 = instances[idx + 1]['observations']['obs']
                action = _preprocess_action(instances[idx + 1]['observations']['action'])
                
                if action == '':
                    continue
                
                template = act_par.parse_action(action)
                if template is None:
                    print('unmatched action: \'{}\''.format(action))
                    action = action
                elif template[0] not in act_par.template2template:
                    if template[0] not in act_par.add_template2template:
                        print('cannot find root: {}'.format(action))
                        action = action
                    else:
                        action = _recover_root_template_action(template, act_par.add_template2template[template[0]])
                else:
                    action = _recover_root_template_action(template, act_par.template2template[template[0]])
                
                positives = []
                candidates = []
                all_actions = instance['valid_actions']
#                 print(all_actions[0])
                if isinstance(all_actions[0], dict):
#                     print(all_actions)
                    all_actions = [all_actions]
                
                for action_group in all_actions:
                    if _match_action(action_group, action):
                        for a in action_group:
                            positives.append(a['a'])
                    else:
                        for a in action_group:
                            candidates.append(a['a'])
                            
                if len(candidates) == 0:
                    continue
                            
                if len(positives) == 0:
                    positives.append(action)
#                     print('adding an action \"{}\" not in valid list'.format(action))
#                     print(all_actions)
#                     if action == 'east':
#                         print(all_actions)

                if idx / len(instances) < 0.6:
                    datasets['train'].add_one(input1, input2, positives, candidates)
                elif idx / len(instances) < 0.8:
                    datasets['dev'].add_one(input1, input2, positives, candidates)
                else:
                    datasets['test'].add_one(input1, input2, positives, candidates)
            
        for k, data_set in datasets.items():
            print('# {} set'.format(k))
            data_set.print_info()

        return datasets
        
        
    def _load_pair_data(self, games, neg_removal=True):
        """
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            positive_pairs -- a list of positive question-passage pairs
            negative_pairs -- a list of negative question-passage pairs
        """
        
        data_set = BERTPair2SeqSet()
        
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
            
        
        def _match_action(action_group, target):
            for action in action_group:
                action = action['a']
                if target == action:
                    return True
            return False
        
        def _process_instance(instance):
            new_inst = json.loads(instance)
            info = new_inst['observations'].split('|')
            new_inst['observations'] = {'origin': new_inst['observations'], 'obs':' | '.join(info[:3]), 'action':info[3]}
            return new_inst
        
        def _recover_root_template_action(template, root_template):
            t_tokens = root_template.split()
            count = 1
            for tid, t_token in enumerate(t_tokens):
                if t_token == 'OBJ':
                    t_tokens[tid] = template[count]
                    count += 1

            return ' '.join(t_tokens)
        
        for game_name in games:
            rom_path = "../roms/jericho-game-suite/{}.z5".format(game_name)
            bindings = load_bindings(rom_path)
            act_par = TemplateActionParser(bindings)
            
            f = open(os.path.join(self.data_dir, '{}.ssa.wt_traj.tok'.format(game_name)), "r")
            instances = f.readlines()
            
#             instances = [json.loads(instance) for instance in instances]
#             for instance in instances:
#                 info = instance['observations'].split('|')
#                 instance['observation'] = {'obs':' | '.join(info[:3]), 'action':info[3]}
            instances = [_process_instance(instance.lower()) for instance in instances]

            for idx, instance in enumerate(instances):
                if idx == len(instances) - 1:
                    continue
            
                input1 = instance['observations']['obs']
                
                input2 = instances[idx + 1]['observations']['obs']
                action = _preprocess_action(instances[idx + 1]['observations']['action'])
                
                template = act_par.parse_action(action)
                if template is None:
                    print('unmatched action: {}'.format(action))
                    action = action
                elif template[0] not in act_par.template2template:
                    if template[0] not in act_par.add_template2template:
                        print('cannot find root: {}'.format(action))
                        action = action
                    else:
                        action = _recover_root_template_action(template, act_par.add_template2template[template[0]])
                else:
                    action = _recover_root_template_action(template, act_par.template2template[template[0]])
                
                positives = []
                candidates = []
                all_actions = instance['valid_actions']
                
                for action_group in all_actions:
                    if _match_action(action_group, action):
                        for a in action_group:
                            positives.append(a['a'])
                    else:
                        for a in action_group:
                            candidates.append(a['a'])
                            
                if len(positives) == 0:
                    positives.append(action)
#                     print('adding an action \"{}\" not in valid list'.format(action))
#                     print(all_actions)
#                     if action == 'east':
#                         print(all_actions)

                data_set.add_one(input1, input2, positives, candidates)
            
        data_set.print_info()

        return data_set
    
    def _numeralize_pairs(self, pairs):
        '''
        numeralize passages in training pair lists
        '''
        ret_pair_list = []
        for pair_dict_ in pairs:
            new_pair_dict_ = {}

            for k, v in pair_dict_.items():
                if k == 'input1' or k == 'input2':
                    new_pair_dict_[k] = self.tokenizer.convert_tokens_to_ids(v.split())
                elif k == 'positives' or k == 'candidates':
                    new_pair_dict_[k] = []
                    for seq in v:
                        new_pair_dict_[k].append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq)))
            ret_pair_list.append(new_pair_dict_)
        return ret_pair_list
    
    
    def _build_vocab(self):
        """
        Filter the vocabulary and numeralization
        """
        
        for data_id, data_set in self.data_sets.items():
            data_set.pairs = self._numeralize_pairs(data_set.get_pairs())

        print('size of the final vocabulary:', len(self.word_vocab))
    
    
    def get_train_batch(self, batch_size, num_negative=-1, inst_format='co_match'):
        """
        randomly select a batch from a dataset
        Inputs:
            batch_size: 
        Outputs:
            q_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            p_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            y_vec -- numpy array of binary labels, numpy array in shape of (batch_size,)
        """
        
        set_id = 'train'
        data_set = self.data_sets[set_id]
#         print(data_set.size())
#         print(batch_size)
        batch_idx = np.random.randint(0, data_set.size(), size=batch_size)
        
        if num_negative < 0:
            num_negative = self.num_negative
        
        if inst_format == 'co_match':
            return self.get_batch(set_id, batch_idx, num_negative)
        elif inst_format == 'concat':
            return self.get_batch_concat(set_id, batch_idx, num_negative)
    
    def get_batch(self, set_id, batch_idx, num_negative=-1):
        """
        randomly select a batch from a dataset
        Inputs:
            batch_idx: 
        Outputs (all numpy arrays are sorted according to q_length):
            q_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            p_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            y_vec -- numpy array of binary labels, numpy array in shape of (batch_size,)
            q_mask -- numpy array of masks
            p_mask -- numpy array of masks
            p_sort_idx -- sorted idx according to p_length
            revert_p_idx -- revert idx from p_mat[p_sort_idx] to p_mat
        """
        
        if num_negative < 0:
            num_negative = self.num_negative
        
        data_set = self.data_sets[set_id]
        x1, x2, positives, candidates, max_x1_len, max_x2_len, max_a_len = data_set.get_samples_from_one_list(batch_idx, 
                                                                                                              num_negative=num_negative, 
                                                                                                              truncate_num=self.truncate_num)
#         qs_, ps_, ys_, max_q_len_, max_p_len_ = data_set.get_samples_from_one_list(batch_idx, self.truncate_num)

        x1_masks_ = []
        x2_masks_ = []
        a_masks_ = [[] for i in range(len(x1))]
        actions = [[] for i in range(len(x1))]
#         print(a_masks_)
#         print(actions)
        for i, q in enumerate(x1):
            x1[i] = q + (max_x1_len - len(q)) * [0]
            x1_masks_.append([1] * len(q) + [0] * (max_x1_len - len(q)))
        for i, p in enumerate(x2):
            x2[i] = p + (max_x2_len - len(p)) * [0]
            x2_masks_.append([1] * len(p) + [0] * (max_x2_len - len(p)))
            
        for i, a in enumerate(positives):
            actions[i].append(a + (max_a_len - len(a)) * [0])
            a_masks_[i].append([1] * len(a) + [0] * (max_a_len - len(a)))
            
        for i, a_list in enumerate(candidates):
            for a in a_list:
                actions[i].append(a + (max_a_len - len(a)) * [0])
                a_masks_[i].append([1] * len(a) + [0] * (max_a_len - len(a)))
            
        x1_mat = np.array(x1, dtype=np.int64)
        x2_mat = np.array(x2, dtype=np.int64)
        x1_mask = np.array(x1_masks_, dtype=np.int64)
        x2_mask = np.array(x2_masks_, dtype=np.int64)
        a_mat = np.array(actions, dtype=np.int64)
        a_mask = np.array(a_masks_, dtype=np.int64)
        y_vec = np.array([0] * len(x1), dtype=np.int64)
        
        return x1_mat, x2_mat, a_mat, y_vec, x1_mask, x2_mask, a_mask
        
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
        x, positives, candidates, max_x_len, max_a_len = data_set.get_concat_samples_from_one_list(batch_idx, 
                                                                                                   num_negative=num_negative, 
                                                                                                   truncate_num=self.truncate_num)

        x_masks_ = []
        a_masks_ = [[] for i in range(len(x))]
        actions = [[] for i in range(len(x))]

        for i, q in enumerate(x):
            x[i] = q + (max_x_len - len(q)) * [0]
            x_masks_.append([1] * len(q) + [0] * (max_x_len - len(q)))
            
        for i, a in enumerate(positives):
            actions[i].append(a + (max_a_len - len(a)) * [0])
            a_masks_[i].append([1] * len(a) + [0] * (max_a_len - len(a)))
            
        for i, a_list in enumerate(candidates):
#             print(len(a_list))
            for a in a_list:
                actions[i].append(a + (max_a_len - len(a)) * [0])
                a_masks_[i].append([1] * len(a) + [0] * (max_a_len - len(a)))
            
        x_mat = np.array(x, dtype=np.int64)
        x_mask = np.array(x_masks_, dtype=np.int64)
        a_mat = np.array(actions, dtype=np.int64)
        a_mask = np.array(a_masks_, dtype=np.int64)
        y_vec = np.array([0] * len(x), dtype=np.int64)
        
        return x_mat, a_mat, y_vec, x_mask, a_mask
    
    def get_eval_batch_concat(self, set_id, inst_idx):
        """
        randomly select a batch from a dataset
        Inputs:
            batch_idx: 
        Outputs (all numpy arrays are sorted according to q_length):
            x_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            a_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            y_list -- numpy array of binary labels, numpy array in shape of (batch_size,)
            x_mask -- numpy array of masks
        """
        
        data_set = self.data_sets[set_id]
        x, candidates, y_list, max_x_len, max_a_len = data_set.get_eval_concat_samples_from_one_list(inst_idx,
                                                                                                   truncate_num=self.truncate_num)

        x_masks_ = []
        a_masks_ = [[] for i in range(len(x))]
        actions = [[] for i in range(len(x))]

        for i, q in enumerate(x):
            x[i] = q + (max_x_len - len(q)) * [0]
            x_masks_.append([1] * len(q) + [0] * (max_x_len - len(q)))
            
        for i, a_list in enumerate(candidates):
#             print(len(a_list))
            for a in a_list:
                actions[i].append(a + (max_a_len - len(a)) * [0])
                a_masks_[i].append([1] * len(a) + [0] * (max_a_len - len(a)))
            
        x_mat = np.array(x, dtype=np.int64)
        x_mask = np.array(x_masks_, dtype=np.int64)
        a_mat = np.array(actions, dtype=np.int64)
        a_mask = np.array(a_masks_, dtype=np.int64)
        
        return x_mat, a_mat, y_list, x_mask, a_mask
    
    
    def display_sentence(self, x):
        """
        Display a suquence of word index
        Inputs:
            x -- input sequence of word indices, (sequence_length,)
        Outputs:
            None
        """
        # apply threshold
        for word_index in x:
            word = self.idx_2_word[word_index]
            if word == '[PAD]':
                continue
            sys.stdout.write(" " + word)
        sys.stdout.write("\n")
        sys.stdout.flush()
        


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
    
    def get_concat_samples_from_one_list(self, batch_idx, num_negative=10, truncate_num=0):
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
            neg_sample_idxs = random.choices(list(range(len(pair_dict_["actions"]))), k=num_negative // 2)
            for idx in neg_sample_idxs:
                action = pair_dict_['actions'][idx]
                next_state = pair_dict_['next_states'][idx]
                
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
                
                
            # sample other tuples with the same action
            neg_samples = random.choices(self.action2tuples[wt_action_str], k=num_negative // 2)
            for (pair_id, idx) in neg_samples:
                if idx >= 0:
                    state = self.pairs[pair_id]['state']
                    next_state = self.pairs[pair_id]['next_states'][idx]
                else:
                    state = self.pairs[pair_id]['state']
                    next_state = self.pairs[pair_id]['wt_next_state']
                
                question = [self.CLS_TOKEN] + state + [self.SEP_TOKEN] + wt_action
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
    
    def get_triple_concat_samples_from_one_list(self, batch_idx, num_negative=10, truncate_num=0):
        concat_triples_list = []
        labels = []
        
        max_lens = []
        
        state_list = []
        action_list = []
        next_state_list = []
        
        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            
            state = pair_dict_['state']
            wt_action = pair_dict_['wt_action']
            wt_action_str = pair_dict_['wt_action_str']
            wt_next_state = pair_dict_['wt_next_state']
            
            state_list.append(state)
            action_list.append(wt_action)
            next_state_list.append(wt_next_state)
            
            # sample other actions under the state
            neg_sample_idxs = random.choices(list(range(len(pair_dict_["actions"]))), k=num_negative // 2)
            for neg_idx in neg_sample_idxs:
                action = pair_dict_['actions'][neg_idx]
                next_state = pair_dict_['next_states'][neg_idx]
                
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                
                
            # sample other tuples with the same action
            neg_samples = random.choices(self.action2tuples[wt_action_str], k=num_negative // 2)
            for (pair_id, neg_idx) in neg_samples:
                if idx >= 0:
                    state = self.pairs[pair_id]['state']
                    next_state = self.pairs[pair_id]['next_states'][neg_idx]
                else:
                    state = self.pairs[pair_id]['state']
                    next_state = self.pairs[pair_id]['wt_next_state']
                
                state_list.append(state)
                action_list.append(wt_action)
                next_state_list.append(next_state)
            
#         print(state_list)
        
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
            
            f = open(os.path.join(self.data_dir, '{}.sas.wt_traj.bert_tok_new'.format(game_name)), "r")
            instances = f.readlines()
            
            instances = [self._process_instance(instance.lower()) for instance in instances]

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
                    
#                     print(next_state)
#                     print(wt_next_state)
                    
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

                rouge_scores = []

                obs_dict = {wt_next_state:1}
                for neg_idx in range(len(actions)):
                    action = actions[neg_idx]
                    next_state = next_states[neg_idx]

                    if action.startswith('drop') and len(action.split(' ')) == 2:
                        continue

                    if next_state not in obs_dict:
                        rouge_score = calc_score([next_state], [state])                            

                        obs_dict[next_state] = 1
                        actions__.append(action)
                        next_states__.append(next_state)
                        rouge_scores.append((len(rouge_scores), rouge_score))

                sorted_rouge_scores = sorted(rouge_scores, key = lambda x:x[1])
                actions_ = []
                next_states_ = []
                for (neg_idx, _) in sorted_rouge_scores:
#                         print(neg_idx)
                    action = actions__[neg_idx]
                    next_state = next_states__[neg_idx]

                    actions_.append(action)
                    next_states_.append(next_state)

                    avg_cand_rouge_l += rouge_score
                    total_cand += 1

                    if len(actions_) == 15:
                        break

                if idx / len(instances) < 0.6:
#                     datasets['train'].add_one(state, next_states, actions, wt_next_state, wt_action)
                    datasets['train'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                else:                                                    
                    if idx / len(instances) < 0.8:
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
        
        for game_name in games + dev_games + test_games:
#             rom_path = "../roms/jericho-game-suite/{}.z5".format(game_name)
            print('# LOADING game data {} ...'.format(game_name))
    
            num_unmatched_wt_action = 0
            
            f = open(os.path.join(self.data_dir, '{}.sas.wt_traj.bert_tok_new'.format(game_name)), "r")
            instances = f.readlines()
            
            instances = [self._process_instance(instance.lower()) for instance in instances]

            for idx, instance in enumerate(instances):
                if idx == len(instances) - 1:
                    continue
            
#                 info = instance['observations']['obs'].split(' | ')
#                 state = ' | '.join([info[0], info[2]])
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
                    
#                     print(next_state)
#                     print(wt_next_state)
                    
                    if next_state == wt_next_state:
                        wt_action = action
                        wt_match_flag = True
                        
                if not wt_match_flag:
#                     print('unmatched action: \'{}\''.format(wt_action_origin))
                    wt_action = wt_action_origin
                    num_unmatched_wt_action += 1
        
#                 next_states_ = next_state
#                 actions_ = actions
        
                rouge_score = calc_score([wt_next_state], [state])
                avg_wt_rouge_l += rouge_score
                total_wt_cand += 1

                actions__ = []
                next_states__ = []

                rouge_scores = []

                obs_dict = {wt_next_state:1}
                for neg_idx in range(len(actions)):
                    action = actions[neg_idx]
                    next_state = next_states[neg_idx]

                    if action.startswith('drop') and len(action.split(' ')) == 2:
                        continue

                    if next_state not in obs_dict:
                        rouge_score = calc_score([next_state], [state])                            

                        obs_dict[next_state] = 1
                        actions__.append(action)
                        next_states__.append(next_state)
                        rouge_scores.append((len(rouge_scores), rouge_score))

                sorted_rouge_scores = sorted(rouge_scores, key = lambda x:x[1])
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
#                     datasets['train'].add_one(state, next_states, actions, wt_next_state, wt_action)
                    datasets['train'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                else:
                    rouge_score = calc_score([wt_next_state], [state])
#                     if idx / len(instances) < 0.5:
                    avg_wt_rouge_l += rouge_score
                    total_wt_cand += 1
                    
                    actions__ = []
                    next_states__ = []
                    
                    rouge_scores = []
                    
                    # TODO: rank actions according to inverse Rouge-L between s and s'
                    
                    obs_dict = {wt_next_state:1}
                    for neg_idx in range(len(actions)):
                        action = actions[neg_idx]
                        next_state = next_states[neg_idx]
                        
                        if action.startswith('drop') and len(action.split(' ')) == 2:
                            continue
                            
                        
                        immed_obs = next_state.split(' | ')[2]
                        if idx / len(instances) >= 0.5:
                            if immed_obs == 'dropped .':
                                if idx % 20 >= 1:
                                    continue
#                             if immed_obs == 'taken .':
#                                 test_num_pick += 1
                            if action == 'burn repellent with torch':
                                if idx % 10 == 1 or idx % 10 == 3 or idx % 10 == 5:
                                    continue
                            if action == 'burn staff with torch':
                                if idx % 10 == 2 or idx % 10 == 3 or idx % 10 == 6:
                                    continue
                        
                        if idx / len(instances) < 0.5:
#                             immed_obs = next_state.split(' | ')[2]
                            if immed_obs == 'dropped .':
                                dev_num_drop += 1
                            if immed_obs == 'taken .':
                                dev_num_pick += 1
                            if action == 'burn repellent with torch' or action == 'burn staff with torch':
                                dev_num_burn += 1
                        else:
#                             immed_obs = next_state.split(' | ')[2]
                            if immed_obs == 'dropped .':
                                test_num_drop += 1
                            if immed_obs == 'taken .':
                                test_num_pick += 1
                            if action == 'burn repellent with torch' or action == 'burn staff with torch':
                                test_num_burn += 1
#                             if action == 'burn repellent with torch':
#                                 continue
#                             if action == 'burn staff with torch':
#                                 continue

                        if next_state not in obs_dict:
                            rouge_score = calc_score([next_state], [state])                            
#                             if rouge_score > 1.0:
#                                 continue
                                
                            obs_dict[next_state] = 1
                            actions__.append(action)
                            next_states__.append(next_state)
                            rouge_scores.append((len(rouge_scores), rouge_score))
                    
#                             avg_cand_rouge_l += rouge_score
#                             total_cand += 1
#                             if len(actions_) == 15:
# #                                 print('================================')
# #                                 for (score, next_state) in zip(rouge_scores, next_states_):
# #                                     print(state)
# #                                     print(next_state)
# #                                     print(score)
# #                                     print('')
#                                 break
                    
                    sorted_rouge_scores = sorted(rouge_scores, key = lambda x:x[1])
                    actions_ = []
                    next_states_ = []
                    for (neg_idx, _) in sorted_rouge_scores:
#                         print(neg_idx)
                        action = actions__[neg_idx]
                        next_state = next_states__[neg_idx]
        
                        immed_obs = next_state.split(' | ')[2]
        
                        if idx / len(instances) < 0.5:
                            if action in dev_act_dict:
                                if immed_obs in dev_act_dict[action]:
#                                     if dev_act_dict[action][immed_obs] == 2:
#                                         continue
                                    dev_act_dict[action][immed_obs] += 1
                                else:
                                    dev_act_dict[action][immed_obs] = 1
                            else:
                                dev_act_dict[action] = {}
                                dev_act_dict[action][immed_obs] = 1
                        else:
#                             if action == 'burn repellent with torch':
#                                 continue
#                             if action == 'burn staff with torch':
#                                 continue
                            if action in test_act_dict:
                                if immed_obs in test_act_dict[action]:
#                                     if test_act_dict[action][immed_obs] == 6:
#                                         continue
                                    test_act_dict[action][immed_obs] += 1
                                else:
                                    test_act_dict[action][immed_obs] = 1
                            else:
                                test_act_dict[action] = {}
                                test_act_dict[action][immed_obs] = 1

                        
                        actions_.append(action)
                        next_states_.append(next_state)
                        
#                         if idx / len(instances) < 0.5:
                        avg_cand_rouge_l += rouge_score
                        total_cand += 1
        
                        if len(actions_) == 15:
                            break

                    if len(test_games) == 0:
                        if idx / len(instances) < 0.5:
                            datasets['dev'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                        else:
                            datasets['test'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                    else:
                        if game_name in dev_games and idx / len(instances) < 0.5:
                            datasets['dev'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                        elif game_name in test_games and idx / len(instances) < 0.5:
                            datasets['test'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
            print('# unmatched actions in the game: {}'.format(num_unmatched_wt_action))
            
        for k, data_set in datasets.items():
            print('# {} set'.format(k))
            data_set.print_info()
        
        print('# averaged rouge-L between walkthrough dev/test (s, s\'): {}'.format(avg_wt_rouge_l/total_wt_cand))
        print('# averaged rouge-L between dev/test (s, s\'): {}'.format(avg_cand_rouge_l/total_cand))
        
        for act_dict in [dev_act_dict, test_act_dict]:
            num_key = 0
            num_bias = 0
            num_unbias = 0
            num_dup = 0
            for k, v in act_dict.items():
                num_key += 1
                num_bias += len(v)
                if len(v) == 1:
                    num_unbias += 1
                    for k2, v2 in v.items():
                        if v2 > 1 and len(act_dict) == len(test_act_dict):
                            print(k + '\t' + k2, '\t', v2)
#                             print('count:', v2)
                        num_dup += v2
                else:
                    if len(act_dict) == len(test_act_dict):
                        print(k, '\t', len(v))
                        for k2, v2 in v.items():
                            print(k2)
            print('num action: {}, num bias: {}, avg bias: {}'.format(num_key, num_bias, num_bias/num_key))
            print('num unbias action: {}, num dup: {}, avg dup: {}'.format(num_unbias, num_dup, num_dup/num_unbias))
            
        print('dev num dropped:', dev_num_drop)
        print('dev num picked:', dev_num_pick)
        print('dev num burn:', dev_num_burn)
        
        print('test num dropped:', test_num_drop)
        print('test num picked:', test_num_pick)
        print('test num burn:', test_num_burn)

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
        concat_inputs, outputs, max_in_len, max_out_len = data_set.get_concat_samples_from_one_list(batch_idx, 
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
        seqs_list, max_lens = data_set.get_triple_concat_samples_from_one_list(batch_idx, 
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
    


# In[ ]:


# def find_game_roms(games, rom_dir):
#     print('#number of games: {}'.format(len(games)))

#     roms = os.listdir(rom_dir)
#     game2rom = {}
#     logs = []
#     for game in games:
#         for rom in roms:
#             if rom.startswith(game + '.z'):
#                 game2rom[game] = rom
#     #             print('find {} for {}'.format(rom, game))
#                 logs.append('find {} for {}'.format(rom, game))
#         if game not in game2rom:
#             print('cannot find rom for {}'.format(game))

#     print('#number of roms founds: {}'.format(len(logs)))
    
#     return game2rom

# if __name__=='__main__':


#     data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/supervised/"
# #     data_dir = "./"
    
#     games = ['905', 'acorncourt', 'advent', 'adventureland', 'afflicted', 'anchor', 'awaken', 
#              'balances', 'deephome', 'detective', 'dragon', 'enchanter', 'gold', 'inhumane', 
#              'jewel', 'karn', 'library', 'ludicorp', 'moonlit', 'omniquest', 'pentari', 'reverb', 
#              'snacktime', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'tryst205', 'yomomma', 
#              'zenon', 'zork1', 'zork3', 'ztuu']
# #     games = ['tryst205', 'yomomma', 
# #              'zenon', 'zork1', 'zork3', 'ztuu']
    
#     games = ['zork1', 'zork3']
#     dev_games = ['zork3']

#     rom_dir = '../roms/jericho-game-suite/'
#     game2rom = find_game_roms(games, rom_dir)
#     print(game2rom)
    
#     pretrain_path = '/dccstor/gaot1/MultiHopReason/comprehension_tasks/narrativeqa/passage_ranker/bert-base-uncased/'
    
#     game_task_data = BERTStateState2ActionDataset(pretrain_path, data_dir, rom_dir=rom_dir, game2rom=game2rom,
#                                               train_games=games, dev_games=games,
#                                               setting='same_games')
# #     game_task_data = StateState2ActionDataset(data_dir, train_games, dev_games)
    
# #     x1, x2, positives, candidates, max_x1_len, max_x2_len, max_a_len = game_task_data.data_sets['train'].get_samples_from_one_list([0,1])

# #     print(x1)
# #     print(x2)
# #     print(positives)
# #     print(candidates)

# #     x1_mat, x2_mat, a_mat, y_vec, x1_mask, x2_mask, a_mask = game_task_data.get_batch('train', [0,1])

# #     print(x1_mat)
# #     print(a_mat)

# #     game_task_data.display_sentence(x1[0])
# #     game_task_data.display_sentence(x1_mat[0])

# #     game_task_data.display_sentence(x2[0])
# #     game_task_data.display_sentence(x2_mat[0])
# #     game_task_data.display_sentence(x2_mat[1])


#     x_mat, a_mat, y_vec, x_mask, a_mask = game_task_data.get_batch_concat('train', [0,1])

#     game_task_data.display_sentence(x_mat[0])
    
#     print(x_mat.shape)
#     print(a_mat.shape)
#     print(x_mask.shape)
#     print(a_mask.shape)
#     print(y_vec.shape)
#     for i in range(a_mat.shape[1]):
#         game_task_data.display_sentence(a_mat[0][i])
    
#     x_mat, a_mat, y_list, x_mask, a_mask = game_task_data.get_eval_batch_concat('train', 1)
#     print(x_mat.shape)
#     print(a_mat.shape)
#     print(x_mask.shape)
#     print(a_mask.shape)
#     print(len(y_list))

#     game_task_data.display_sentence(x_mat[0])
#     for i in range(a_mat.shape[1]):
#         game_task_data.display_sentence(a_mat[0][i])
#     print(y_list)


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


    data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/zork_universe_sup/"
#     data_dir = "./"
    
    
    games = ['zork1', 'zork3']
    dev_games = ['zork3']
#     games = ['zork1', 'zork3', 'enchanter', 'spellbrkr', 'sorcerer']
    games = ['zork1', 'zork3', 'enchanter', 'sorcerer']
    train_games = ['zork1', 'enchanter', 'sorcerer']
    dev_games = ['zork3']
#     games = ['spellbrkr']
    
    rom_dir = '../roms/jericho-game-suite/'
    game2rom = find_game_roms(games, rom_dir)
    print(game2rom)
    
    pretrain_path = '/dccstor/gaot1/MultiHopReason/comprehension_tasks/narrativeqa/passage_ranker/bert-base-uncased/'
    
#     game_task_data = BERTStateAction2StateDataset(pretrain_path, data_dir, rom_dir=rom_dir, game2rom=game2rom,
#                                               train_games=games, dev_games=games,
#                                               setting='same_games')

    game_task_data = BERTStateAction2StateDataset(pretrain_path, data_dir, rom_dir=rom_dir, game2rom=game2rom,
                                              train_games=train_games, dev_games=dev_games,
                                              setting='transfer')

    game_task_data.data_sets['dev'].check_eval_triples(game_task_data.idx_2_word)
    game_task_data.data_sets['test'].check_eval_triples(game_task_data.idx_2_word)

    
    i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_batch_concat('train', [4,5], num_negative=4)

    for i in range(i_mat.shape[0]):
        game_task_data.display_sentence(i_mat[i])
        game_task_data.display_sentence(o_mat[i])
    #     print(concat_inputs[i])
    #     print(outputs[i])
        print('')
        
    print(y_vec)
    print(i_mat.shape)
    
    # TODO: compute which action has most occurrance of tuples, 
    # then the groundtruth action use the most frequent one so the second negative sampling makes more sense

    concat_inputs, outputs, max_in_len, max_out_len = game_task_data.data_sets['train'].get_concat_samples_from_one_list([0,1], num_negative=4)

    print(len(concat_inputs))

    for i in range(len(concat_inputs)):
        game_task_data.display_sentence(concat_inputs[i])
        game_task_data.display_sentence(outputs[i])
    #     print(concat_inputs[i])
    #     print(outputs[i])
        print('')
    # game_task_data.display_sentence(concat_inputs[1])
    # game_task_data.display_sentence(outputs[1])
    
    mat_list, y_vec, masks_list = game_task_data.get_batch_triple_concat('train', [4,5], num_negative=4)

    print(len(mat_list))
    print(len(masks_list))
    print(y_vec)

    print(mat_list[0])
    print(masks_list[0])

    j=5
    for i in range(len(mat_list[j])):
        print(len(mat_list[j][i]))
        game_task_data.display_sentence(mat_list[j][i])


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


# * no filtering
# 
#         # averaged rouge-L between dev/test (s, s): 0.6941369171333641
#         size of the final vocabulary: 30522
#         max number of cand: 27
#         avg number of cand: 1759/136=12.933823529411764
#         max number of cand: 25
#         avg number of cand: 1831/135=13.562962962962963
#         
#         dev acc: 0.157256, best dev acc: 0.157256
#         test acc: 0.158204
# 
# * first-15 actions:
# 
#         # averaged rouge-L between dev/test (s, s): 0.7202821974849056
#         max number of cand: 15
#         avg number of cand: 1432/136=10.529411764705882
#         max number of cand: 15
#         avg number of cand: 1759/135=13.02962962962963
#         
#         dev acc: 0.199617, best dev acc: 0.199617
#         test acc: 0.199098
#         
# * inverse-rank top-15 actions:
# 
#         # averaged rouge-L between dev/test (s, s): 0.35079739614933614
#         max number of cand: 15
#         avg number of cand: 1432/136=10.529411764705882
#         max number of cand: 15
#         avg number of cand: 1759/135=13.02962962962963
#         
#         
#     #### removing "drop sth"
#         # averaged rouge-L between walkthrough dev/test (s, s'): 0.6805402445992036
#         # averaged rouge-L between dev/test (s, s'): 0.3323068400857799
#         max number of cand: 15
#         avg number of cand: 1140/136=8.382352941176471
#         max number of cand: 15
#         avg number of cand: 1438/135=10.651851851851852
# 
# * rouge <= 0.8 actions:
# 
#         # averaged rouge-L between dev/test (s, s): 0.5303684307968914
#         max number of cand: 24
#         avg number of cand: 1055/136=7.757352941176471
#         max number of cand: 19
#         avg number of cand: 881/135=6.525925925925926
# 
# * rouge <= 0.75 actions:
# 
#         # averaged rouge-L between dev/test (s, s): 0.4584714476524355
#         max number of cand: 24
#         avg number of cand: 916/136=6.735294117647059
#         max number of cand: 19
#         avg number of cand: 581/135=4.303703703703704

# 
#     # train set
#     Number of walkthrough tuples: 715
#     Number of tuples: 12350
#     Number of unique actions: 1229
#     # dev set
#     Number of walkthrough tuples: 238
#     Number of tuples: 3217
#     Number of unique actions: 379
#     # test set
#     Number of walkthrough tuples: 231
#     Number of tuples: 2629
#     Number of unique actions: 444
#     size of the final vocabulary: 30522
#     max number of cand: 15
#     avg number of cand: 3217/238=13.516806722689076
#     max number of cand: 15
#     avg number of cand: 2629/231=11.380952380952381
#     
#     # train set
#     Number of walkthrough tuples: 715
#     Number of tuples: 12350
#     Number of unique actions: 1229
#     # dev set
#     Number of walkthrough tuples: 238
#     Number of tuples: 4758
#     Number of unique actions: 566
#     # test set
#     Number of walkthrough tuples: 231
#     Number of tuples: 3518
#     Number of unique actions: 573
#     size of the final vocabulary: 30522
#     max number of cand: 45
#     avg number of cand: 3998/238=16.798319327731093
#     max number of cand: 46
#     avg number of cand: 3037/231=13.147186147186147
# 
# 
#      399 zork1.sas.wt_traj.txt
#      272 zork3.sas.wt_traj.txt
#      264 enchanter.sas.wt_traj.txt
#      202 spellbrkr.sas.wt_traj.txt
#      253 sorcerer.sas.wt_traj.txt
# 
# 
# # LOADING game data zork1 ...
# # unmatched actions in the game: 36
# # LOADING game data zork3 ...
# # unmatched actions in the game: 140
# # LOADING game data enchanter ...
# # unmatched actions in the game: 102
# # LOADING game data spellbrkr ...
# # unmatched actions in the game: 198
# # LOADING game data sorcerer ...
# # unmatched actions in the game: 99
# 
#     # train set
#     Number of walkthrough tuples: 913
#     Number of tuples: 16828
#     Number of unique actions: 1619
#     # dev set
#     Number of walkthrough tuples: 136
#     Number of tuples: 1432
#     Number of unique actions: 143
#     # test set
#     Number of walkthrough tuples: 135
#     Number of tuples: 1759
#     Number of unique actions: 126
#     size of the final vocabulary: 30522
#     max number of cand: 15
#     avg number of cand: 1432/136=10.529411764705882
#     max number of cand: 15
#     avg number of cand: 1759/135=13.02962962962963
# 
#     # train set
#     Number of walkthrough tuples: 913
#     Number of tuples: 16828
#     Number of unique actions: 1619
#     # dev set
#     Number of walkthrough tuples: 136
#     Number of tuples: 1846
#     Number of unique actions: 155
#     # test set
#     Number of walkthrough tuples: 135
#     Number of tuples: 1952
#     Number of unique actions: 156
#     size of the final vocabulary: 30522

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




