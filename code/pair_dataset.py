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

import random


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


class Seq2SeqSet(object):
    '''
    self.positive_pairs
    self.negative_pairs
    '''
    def __init__(self):
        self.pairs = []
        self.positive_dict = {}
        self.negative_dict = {}
        
    def add_one(self, observation, positives, candidates):
        self.pairs.append({'observation':observation, 'positives':positives, 'candidates':candidates})
        if label == 0:
            self.positive_dict[len(self.pairs)] = 1
        else:
            self.negative_dict[len(self.pairs)] = 1
        
    def get_pairs(self):
        return self.pairs
    
    def size(self):
        return len(self.pairs)
    
    def get_samples_from_one_list(self, batch_idx, truncate_num=0):
        qs = []
        ps = []
        ys = []
        max_q_len = -1
        max_p_len = -1
        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = pair_dict_['label']
            ys.append(label)
            
            question = pair_dict_['hypothesis']
            
            if truncate_num > 0:
                question = question[:truncate_num]
            if len(question) > max_q_len:
                max_q_len = len(question)
                
            qs.append(question)

            passage = pair_dict_['premise']
                
            if truncate_num > 0:
                passage = passage[:truncate_num]
            if len(passage) > max_p_len:
                max_p_len = len(passage)
                
            ps.append(passage)

            
        return qs, ps, ys, max_q_len, max_p_len
            
    def print_info(self):
        print('Number of positive pairs:', len(self.positive_dict))
        print('Number of negative pairs:', len(self.negative_dict))


# In[ ]:


class State2ActionDataset(TextDataset):
    
    def __init__(self, data_dir, truncate_num=300, freq_threshold=2):        
        super(State2ActionDataset, self).__init__(data_dir)
        self.truncate_num = truncate_num
        self.freq_threshold = freq_threshold
        
        self.word_vocab = {'<PAD>':0, '<START>':1, '<END>':2, '<UNK>':3, '<ANSWER>':4, '<SPLIT>':5, ',':6}
        self.label_vocab = {'entails':0, 'neutral':1}
        self.load_dataset()
        print('Converting text to word indicies.')
        self.idx_2_word = self._index_to_word()
        
        
    def load_dataset(self):
        
        self.data_sets = {}
        
        # load train
        self.data_sets['train'] = self._load_pair_data(os.path.join(self.data_dir, 'train.tsv'))
#         self.data_sets['train'] = self._load_pair_data(os.path.join(self.data_dir, 'dev.tsv'))
#         self.data_sets['train'] = self._load_pair_data(os.path.join(self.data_dir, 
#                                                                     'scitail_1.0_structure_subset_triple.train.tsv'))
        
        # load dev
        self.data_sets['dev'] = self._load_pair_data(os.path.join(self.data_dir, 'dev.tsv'))
        self.data_sets['test'] = self._load_pair_data(os.path.join(self.data_dir, 'test.tsv'))
#         self.data_sets['dev'] = self._load_pair_data(os.path.join(self.data_dir, 'scitail_1.0_structure_subset_triple.dev.tsv'))
        
        # build vocab
        self._build_vocab()
        
        
    def _load_pair_data(self, fpath, neg_removal=True):
        """
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            positive_pairs -- a list of positive question-passage pairs
            negative_pairs -- a list of negative question-passage pairs
        """
        
        data_set = PairClassificationSet()
        
        f = open(fpath, "r")
        instances = f.readlines()
        
        for idx, instance in enumerate(instances):
            [premise, hypothesis, label] = instance.strip('\n').lower().split('\t')
            label = self.label_vocab[label]
            
            data_set.add_one(hypothesis, premise, label)
            
        data_set.print_info()

        return data_set
    
    def _numeralize_pairs(self, word_freq_dict, pairs):
        '''
        numeralize passages in training pair lists
        '''
        ret_pair_list = []
        for pair_dict_ in pairs:
            new_pair_dict_ = {}

            for k, v in pair_dict_.items():
                if k != 'label':
                    new_pair_dict_[k] = self._add_vocab_from_sentence(word_freq_dict, v)
                else:
                    new_pair_dict_[k] = pair_dict_[k] 

            ret_pair_list.append(new_pair_dict_)
        return ret_pair_list
    
    
    def _add_vocab_from_sentence(self, word_freq_dict, sentence):
        tokens = sentence.split(' ')
        word_idx_list = []
        for token in tokens:
            if word_freq_dict[token] < self.freq_threshold:
                word_idx_list.append(self.word_vocab['<UNK>'])
            else:
                if token not in self.word_vocab:
                    self.word_vocab[token] = len(self.word_vocab)
                word_idx_list.append(self.word_vocab[token])
        return word_idx_list
    
    
    def _build_vocab(self):
        """
        Filter the vocabulary and numeralization
        """
        
        word_freq_dict = self._get_word_freq(self.data_sets)
            
        for data_id, data_set in self.data_sets.items():
            data_set.pairs = self._numeralize_pairs(word_freq_dict, data_set.get_pairs())

        print('size of the final vocabulary:', len(self.word_vocab))
        
        
    def _add_freq_from_sentence(self, word_freq_dict, sentence):
        tokens = sentence.split(' ')
        for token in tokens:
            if token not in word_freq_dict:
                word_freq_dict[token] = 1
            else:
                word_freq_dict[token] += 1
        
    def _get_word_freq(self, data_sets_):
        """
        Building word frequency dictionary and filter the vocabulary
        """

        word_freq_dict = {}

        for data_id, data_set in data_sets_.items():
            for pair_dict in data_set.get_pairs():
                for sentence in [pair_dict['hypothesis'], pair_dict['premise']]:
                    self._add_freq_from_sentence(word_freq_dict, sentence)

        print('size of the raw vocabulary:', len(word_freq_dict))
        return word_freq_dict
    
    
    def get_train_batch(self, batch_size, sort=False):
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
        batch_idx = np.random.randint(0, data_set.size(), size=batch_size)
        
        return self.get_batch(set_id, batch_idx, sort)
    
    def get_batch(self, set_id, batch_idx, sort=False):
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
        
        data_set = self.data_sets[set_id]
        qs_, ps_, ys_, max_q_len_, max_p_len_ = data_set.get_samples_from_one_list(batch_idx, self.truncate_num)

        q_masks_ = []
        p_masks_ = []
        for i, q in enumerate(qs_):
            qs_[i] = q + (max_q_len_ - len(q)) * [0]
            q_masks_.append([1] * len(q) + [0] * (max_q_len_ - len(q)))
        for i, p in enumerate(ps_):
            ps_[i] = p + (max_p_len_ - len(p)) * [0]
            p_masks_.append([1] * len(p) + [0] * (max_p_len_ - len(p)))
            
        q_mat = np.array(qs_, dtype=np.int64)
        p_mat = np.array(ps_, dtype=np.int64)
        q_mask = np.array(q_masks_, dtype=np.int64)
        p_mask = np.array(p_masks_, dtype=np.int64)        
        y_vec = np.array(ys_, dtype=np.int64)
        
        if sort:
            # sort all according to q_length
            q_length = np.sum(q_mask, axis=1)
            q_sort_idx = np.argsort(-q_length)
            q_mat = q_mat[q_sort_idx, :]
            p_mat = p_mat[q_sort_idx, :]
            q_mask = q_mask[q_sort_idx, :]
            p_mask = p_mask[q_sort_idx, :]
            y_vec = y_vec[q_sort_idx]

            # get p_sorted_idx and the revert idx
            p_length = np.sum(p_mask, axis=1)
            p_sort_idx = np.argsort(-p_length)
            idx_dict = {p_sort_idx[i_]: i_ for i_ in range(p_length.shape[0])}
            revert_p_idx = np.array([ idx_dict[i_] for i_ in range(p_length.shape[0])])
        
            return q_mat, p_mat, y_vec, q_mask, p_mask, p_sort_idx, revert_p_idx

        else:
            return q_mat, p_mat, y_vec, q_mask, p_mask
    
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
            if word == '<PAD>':
                continue
            sys.stdout.write(" " + word)
        sys.stdout.write("\n")
        sys.stdout.flush()
        
    def initial_conceptnet_embedding(self, embedding_size, embedding_path=None):
        """
        This function initialize embedding with glove embedding. 
        If a word has embedding in glove, use the glove one.
        If not, initial with random.
        Inputs:
            embedding_size -- the dimension of the word embedding
            embedding_path -- the path to the glove embedding file
        Outputs:
            embeddings -- a numpy matrix in shape of (vocab_size, embedding_dim)
                          the ith row indicates the word with index i from word_ind_dict
        """    
        vocab_size = len(self.word_vocab)
        # initialize a numpy embedding matrix 
        embeddings = 0.1*np.random.randn(vocab_size, embedding_size).astype(np.float32)
        # replace <PAD> by all zero
        embeddings[0, :] = np.zeros(embedding_size, dtype=np.float32)

        if embedding_path and os.path.isfile(embedding_path):
            f = open(embedding_path, "r")
            counter = 0
            f.readline()
            for line in f:
                data = line.strip().split(" ")
                word = data[0].strip()
                embedding = data[1::]
                embedding = map(np.float32, embedding)
                if word in self.word_vocab:
                    embeddings[self.word_vocab[word], :] = embedding
                    counter += 1
            f.close()
            print("%d words has been switched." %counter)
        else:
            print("embedding is initialized fully randomly.")
            
        return embeddings


# In[ ]:


import re
from jericho.util import verb_usage_count
from jericho.template_action_generator import TemplateActionGenerator

class TemplateActionParser(TemplateActionGenerator):
    def __init__(self, rom_bindings):        
        self.templates_alias_dict = {}
        self.verb_to_templates = {}
        self.template2template = {}
        super(TemplateActionParser, self).__init__(rom_bindings)
        
        self.id2template = None
        self.template2id = None
        
        self.additional_templates = ['land']
        self.templates = list(set(self.templates + self.additional_templates))

        self.templates.sort()
        self._compute_template()
        
        BASIC_ACTIONS = 'north/south/west/east/northwest/southwest/northeast/southeast/up/down/enter/exit/take all'.split('/')
        self.BASIC_ACTIONS = {k:1 for k in BASIC_ACTIONS}
        
        self.add_template2template = {}
        for action in list(self.BASIC_ACTIONS.keys()) + self.additional_templates + ['examine OBJ']:
            self.add_template2template[action] = action
        
        
    def _preprocess_templates(self, templates, max_word_length):
        '''
        Converts templates with multiple verbs and takes the first verb.
        '''
        out = []
        vb_usage_fn = lambda verb: verb_usage_count(verb, max_word_length)
        p = re.compile('\S+(/\S+)+')
        for template in templates:
#             print(template)
            if not template:
                continue
            has_alias = True
            while True:
                match = p.search(template)
                if not match:
#                     print('{} not matched'.format(template))
                    has_alias = False
                    break
                    
                verb_alias = match.group().split('/')
                
                verb = max(match.group().split('/'), key=vb_usage_fn)
                verb_template = template[:match.start()] + verb + template[match.end():]
                
                for alias in verb_alias:
                    alias_template = template[:match.start()] + alias + template[match.end():]
                    self.template2template[alias_template] = verb_template
                    
                    if alias in self.verb_to_templates:
                        self.verb_to_templates[alias].append(alias_template)
                    else:
                        self.verb_to_templates[alias] = [alias_template]
                
#                 for alias in verb_alias:
#                     if alias in self.verb_to_templates:
#                         self.verb_to_templates[alias].append(template)
#                     else:
#                         self.verb_to_templates[alias] = [template]
                template = verb_template
                
            ts = template.split()
            if ts[0] in defines.ILLEGAL_ACTIONS:
                continue
            if ts[0] in defines.NO_EFFECT_ACTIONS and len(ts) == 1:
                continue
                
            if not has_alias:
                t_tokens = template.split()
                alias = t_tokens[0]
                verb_alias = [alias]
                if alias in self.verb_to_templates:
                    self.verb_to_templates[alias].append(template)
                else:
                    self.verb_to_templates[alias] = [template]
                    
                self.template2template[template] = template
                
            self.templates_alias_dict[template] = verb_alias
            out.append(template)
        return out
    
    def _compute_template(self):
        self.id2template = {}
        self.template2id = {}
        for i, t in enumerate(self.templates):
            self.id2template[i] = t
            self.template2id[t] = i
        return

    def parse_action(self, action):

        tokens = action.split()
        verb = tokens[0]
#         if verb == 'down':
#             print(verb in self.BASIC_ACTIONS and len(tokens) == 1)

        if (verb in self.BASIC_ACTIONS or verb in self.additional_templates) and len(tokens) == 1:
            return [verb]

        if verb not in self.verb_to_templates:
#             if (verb in self.BASIC_ACTIONS or verb in self.additional_templates) and len(tokens) == 1:
#     #             print(verb)
#                 return [verb]
            if verb == 'examine':
                return ['examine OBJ', ' '.join(tokens[1:])]
            else:
                print('cannot recognize verb:', verb)
                return None
        else:
            templates = self.verb_to_templates[verb]
            for template in templates:
#                 print(template.split())
                t_tokens = template.split()
#                 print(t_tokens)
                
                slot_num = 0
                for t_token in t_tokens:
#                     print(t_token, 'OBJ', t_token == 'OBJ')
                    if t_token == 'OBJ':
                        slot_num += 1
#                 ' \S+'
                re_str = template.replace('OBJ', '(\S+)')
    #             print(re_str)
    #             p = re.compile('\S+(/\S+)+')
                p = re.compile(re_str)

                match = p.search(action)
                if not match:
                    continue
                elif match.group() == action:
                    ret_tuple = [template]
#                     print(slot_num)
                    for i in range(slot_num):
                        ret_tuple.append(match.group(i+1))
                    return ret_tuple
                else:
                    continue
        
        templates = self.verb_to_templates[verb]
        for template in templates:
            t_tokens = template.split()
            slot_num = 0
            for t_id, t_token in enumerate(t_tokens):
                if t_token == 'OBJ':
                    slot_num += 1
                    t_tokens[t_id] = 'OBJ%d'%(slot_num - 1)
#                 ' \S+'

            re_str = ' '.join(t_tokens)
            for i in range(slot_num):
                re_str = re_str.replace('OBJ%d'%(i), '(?P<obj%d>\S+( \S+)*)'%(i))
#             print(re_str)
#             p = re.compile('\S+(/\S+)+')
            p = re.compile(re_str)

            match = p.search(action)
            if not match:
                continue
            elif match.group() == action:
                ret_tuple = [template]
                for i in range(slot_num):
                    ret_tuple.append(match.group('obj%d'%(i)))
                return ret_tuple
            else:
                continue
        return None   
    
# act_par = TemplateActionParser(bindings)
# print(act_par.templates_alias_dict)
# print(act_par.verb_to_templates)


# In[ ]:


class Pair2SeqSet(object):
    '''
    '''
    def __init__(self):
        self.pairs = []
        self.num_positive = 0
        self.num_total = 0
        
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
            if len(label) > max_a_len:
                max_a_len = len(label)
            positives.append(label)
            
            cand_list = []
#             num_neg_samples = min(len(pair_dict_["candidates"]), num_negative)
            neg_samples = random.choices(pair_dict_["candidates"], k=num_negative)
            for neg_sample in neg_samples:
                if len(neg_sample) > max_a_len:
                    max_a_len = len(neg_sample)
                cand_list.append(neg_sample)
            candidates.append(cand_list)
            
            question = pair_dict_['input1']
            if truncate_num > 0:
                question = question[:truncate_num]
            if len(question) > max_x1_len:
                max_x1_len = len(question)
                
            x1.append(question)

            passage = pair_dict_['input2']
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
            if len(label) > max_a_len:
                max_a_len = len(label)
            positives.append(label)
            
            cand_list = []
#             num_neg_samples = min(len(pair_dict_["candidates"]), num_negative)
#             print('sampling {} actions'.format(num_negative))
            neg_samples = random.choices(pair_dict_["candidates"], k=num_negative)
#             print('{} actions sampled'.format(len(neg_samples)))
            for neg_sample in neg_samples:
                if len(neg_sample) > max_a_len:
                    max_a_len = len(neg_sample)
                cand_list.append(neg_sample)
            candidates.append(cand_list)
#             print(len(cand_list))
            
            question = pair_dict_['input1'] + [5] + pair_dict_['input2']
            
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
            if len(action) > max_a_len:
                max_a_len = len(action)
            cand_list.append(action)
            y_list.append(1)
            positive_dict[_get_key_from_list(action)] = 1
            
        for action in pair_dict_["candidates"]:
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
        if truncate_num > 0:
            question = question[:truncate_num]
        if len(question) > max_x1_len:
            max_x1_len = len(question)

        x1.append(question)

        passage = pair_dict_['input2']
        if truncate_num > 0:
            passage = passage[:truncate_num]
        if len(passage) > max_x2_len:
            max_x2_len = len(passage)

        x2.append(passage)
            
        return x1, x2, candidates, y_list, max_x1_len, max_x2_len, max_a_len
    
    def get_eval_concat_samples_from_one_list(self, inst_idx, truncate_num=0):
        concat_x = []
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
            if len(action) > max_a_len:
                max_a_len = len(action)
            cand_list.append(action)
            y_list.append(1)
            positive_dict[_get_key_from_list(action)] = 1
        
        for action in pair_dict_["candidates"]:
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

        question = pair_dict_['input1'] + [5] + pair_dict_['input2']

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


def _tokenize_original_data(games, data_dir):
    
    import spacy
#     nlp_pipe = spacy.blank("en")
    nlp_pipe = spacy.load('en')
    
    def _tokenize_observation(instance):
        
#         doc = nlp_pipe(line.strip())
        
        new_inst = json.loads(instance)
        info = new_inst['observations'].split('|')
        for i in range(3):
            doc = nlp_pipe(info[i])
            text_tokens = [token.text for token in doc if token.text != ' ']
            info[i] = ' '.join(text_tokens)
        new_inst['observations'] = '|'.join(info)
        
        return new_inst
    
    for game_name in games:
        print('# LOADING game data {} ...'.format(game_name))

        f = open(os.path.join(data_dir, '{}.ssa.wt_traj.txt'.format(game_name)), "r")
        instances = f.readlines()
        
        fout = open(os.path.join(data_dir, '{}.ssa.wt_traj.tok'.format(game_name)), "w")

        for idx, instance in enumerate(instances):
            instance_tok = _tokenize_observation(instance)
            fout.write(json.dumps(instance_tok) + '\n')
            
        f.close()
        fout.close()
            
# data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/supervised/"

# games = ['905', 'acorncourt', 'advent', 'adventureland', 'afflicted', 'anchor', 'awaken', 
#          'balances', 'deephome', 'detective', 'dragon', 'enchanter', 'gold', 'inhumane', 
#          'jewel', 'karn', 'library', 'ludicorp', 'moonlit', 'omniquest', 'pentari', 'reverb', 
#          'snacktime', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'tryst205', 'yomomma', 
#          'zenon', 'zork1', 'zork3', 'ztuu']

# _tokenize_original_data(games, data_dir)
            


# In[ ]:


def _tokenize_original_sas_data(games, data_dir):
    
    import spacy
    nlp_pipe = spacy.load('en')
    
    def _tokenize_observation(instance):
        
#         doc = nlp_pipe(line.strip())
        
        new_inst = json.loads(instance)
        info = new_inst['observations'].split('|')
        for i in range(3):
            doc = nlp_pipe(info[i])
            text_tokens = [token.text for token in doc if token.text != ' ']
            info[i] = ' '.join(text_tokens)
        new_inst['observations'] = '|'.join(info)
        
        for idx, action_group in enumerate(new_inst['valid_actions']):
            action_tuple = action_group[0]
            info = action_tuple['observations'].split('|')
            for i in range(3):
                doc = nlp_pipe(info[i])
                text_tokens = [token.text for token in doc if token.text != ' ']
                info[i] = ' '.join(text_tokens)
            new_inst['valid_actions'][idx][0]['observations'] = '|'.join(info)
        
        return new_inst
    
    for game_name in games:
        print('# LOADING game data {} ...'.format(game_name))

        f = open(os.path.join(data_dir, '{}.sas.wt_traj.txt'.format(game_name)), "r")
        instances = f.readlines()
        
        fout = open(os.path.join(data_dir, '{}.sas.wt_traj.tok'.format(game_name)), "w")

        for idx, instance in enumerate(instances):
            instance_tok = _tokenize_observation(instance)
            fout.write(json.dumps(instance_tok) + '\n')
            
        f.close()
        fout.close()

# games = ['zork1', 'zork3', 'enchanter', 'sorcerer']
# data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/zork_universe_sup/"

# _tokenize_original_sas_data(games, data_dir)
            


# In[ ]:


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
    new_inst['observations'] = {'obs':' | '.join(info[:3]), 'action':info[3]}
    return new_inst

def _recover_root_template_action(template, root_template):
    t_tokens = root_template.split()
    count = 1
    for tid, t_token in enumerate(t_tokens):
        if t_token == 'OBJ':
            t_tokens[tid] = template[count]
            count += 1

    return ' '.join(t_tokens)

class StateState2ActionDataset(TextDataset):
    
    def __init__(self, data_dir, rom_dir, game2rom, train_games=None, dev_games=None, setting='same_games',
                 num_negative=20, truncate_num=300, freq_threshold=2):        
        super(StateState2ActionDataset, self).__init__(data_dir)
        
        self.num_negative = num_negative
        self.truncate_num = truncate_num
        self.freq_threshold = freq_threshold
        
        self.word_vocab = {'<PAD>':0, '<START>':1, '<END>':2, '<UNK>':3, '<ANSWER>':4, '<SPLIT>':5, '|':6}
        
        self.rom_dir = rom_dir
        self.game2rom = game2rom
        self.setting = setting
        self.train_games = train_games
        self.dev_games = dev_games
        
        self.load_dataset()
        print('Converting text to word indicies.')
        self.idx_2_word = self._index_to_word()
        
        
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
        datasets['train'] = Pair2SeqSet()
        datasets['dev'] = Pair2SeqSet()
        datasets['test'] = Pair2SeqSet()
        
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
        
        data_set = Pair2SeqSet()
        
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
            new_inst['observations'] = {'obs':' | '.join(info[:3]), 'action':info[3]}
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
    
    def _numeralize_pairs(self, word_freq_dict, pairs):
        '''
        numeralize passages in training pair lists
        '''
        ret_pair_list = []
        for pair_dict_ in pairs:
            new_pair_dict_ = {}

            for k, v in pair_dict_.items():
                if k == 'input1' or k == 'input2':
                    new_pair_dict_[k] = self._add_vocab_from_sentence(word_freq_dict, v)
                elif k == 'positives' or k == 'candidates':
                    new_pair_dict_[k] = []
                    for seq in v:
                        new_pair_dict_[k].append(self._add_vocab_from_sentence(word_freq_dict, seq))

            ret_pair_list.append(new_pair_dict_)
        return ret_pair_list
    
    
    def _add_vocab_from_sentence(self, word_freq_dict, sentence):
        tokens = sentence.split(' ')
        word_idx_list = []
        for token in tokens:
            if word_freq_dict[token] < self.freq_threshold:
                word_idx_list.append(self.word_vocab['<UNK>'])
            else:
                if token not in self.word_vocab:
                    self.word_vocab[token] = len(self.word_vocab)
                word_idx_list.append(self.word_vocab[token])
        return word_idx_list
    
    
    def _build_vocab(self):
        """
        Filter the vocabulary and numeralization
        """
        
        word_freq_dict = self._get_word_freq(self.data_sets)
            
        for data_id, data_set in self.data_sets.items():
            data_set.pairs = self._numeralize_pairs(word_freq_dict, data_set.get_pairs())

        print('size of the final vocabulary:', len(self.word_vocab))
        
        
    def _add_freq_from_sentence(self, word_freq_dict, sentence):
        tokens = sentence.split(' ')
        for token in tokens:
            if token not in word_freq_dict:
                word_freq_dict[token] = 1
            else:
                word_freq_dict[token] += 1
        
    def _get_word_freq(self, data_sets_):
        """
        Building word frequency dictionary and filter the vocabulary
        """

        word_freq_dict = {}

        for data_id, data_set in data_sets_.items():
            for pair_dict in data_set.get_pairs():
                for sentence in [pair_dict['input1'], pair_dict['input2']]:
                    self._add_freq_from_sentence(word_freq_dict, sentence)
                for sentence in pair_dict['positives'] + pair_dict['candidates']:
                    self._add_freq_from_sentence(word_freq_dict, sentence)

        print('size of the raw vocabulary:', len(word_freq_dict))
        return word_freq_dict
    
    
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
    
    def get_eval_batch(self, set_id, inst_idx):
        
        data_set = self.data_sets[set_id]
        
        x1, x2, candidates, y_list, max_x1_len, max_x2_len, max_a_len = data_set.get_eval_samples_from_one_list(inst_idx, 
                                                                                                              truncate_num=self.truncate_num)
#         qs_, ps_, ys_, max_q_len_, max_p_len_ = data_set.get_samples_from_one_list(batch_idx, self.truncate_num)

        x1_masks_ = []
        x2_masks_ = []
        a_masks_ = [[] for i in range(len(x1))]
        actions = [[] for i in range(len(x1))]

        for i, q in enumerate(x1):
            x1[i] = q + (max_x1_len - len(q)) * [0]
            x1_masks_.append([1] * len(q) + [0] * (max_x1_len - len(q)))
        for i, p in enumerate(x2):
            x2[i] = p + (max_x2_len - len(p)) * [0]
            x2_masks_.append([1] * len(p) + [0] * (max_x2_len - len(p)))
            
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
        
        return x1_mat, x2_mat, a_mat, y_list, x1_mask, x2_mask, a_mask
    
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
            if word == '<PAD>':
                continue
            sys.stdout.write(" " + word)
        sys.stdout.write("\n")
        sys.stdout.flush()
        


# In[ ]:


class ForwardPredictionSet(object):
    '''
    '''
    def __init__(self):
        self.pairs = []
        self.num_positive = 0
        self.num_tuples = 0
        
        self.SEP_TOKEN = 5
        
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

    def get_triple_samples(self, batch_idx, num_negative=10, truncate_num=0):
        states = []
        actions = []
        outputs = []
        labels = []
        positives = []
        candidates = []
        
        max_in_len = -1
        max_a_len = -1
        max_out_len = -1
        
        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            
            state = pair_dict_['state']
            wt_action = pair_dict_['wt_action']
            wt_action_str = pair_dict_['wt_action_str']
            wt_next_state = pair_dict_['wt_next_state']
            
            wt_question = state
            if truncate_num > 0:
                wt_question = wt_question[:truncate_num]
            if len(wt_question) > max_in_len:
                max_in_len = len(wt_question)
            states.append(wt_question)
            
            if len(wt_action) > max_a_len:
                max_a_len = len(wt_action)
            actions.append(wt_action)
            
            wt_next_state = wt_next_state
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
                
                question = state
                if truncate_num > 0:
                    question = question[:truncate_num]
                if len(question) > max_in_len:
                    max_in_len = len(question)
                states.append(question)
                
                if len(action) > max_a_len:
                    max_a_len = len(action)
                actions.append(action)

                next_state = next_state
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
                
                question = state
                if truncate_num > 0:
                    question = question[:truncate_num]
                if len(question) > max_in_len:
                    max_in_len = len(question)
                states.append(question)
                
                if len(wt_action) > max_a_len:
                    max_a_len = len(wt_action)
                actions.append(wt_action)

                if truncate_num > 0:
                    next_state = next_state[:truncate_num]
                if len(next_state) > max_out_len:
                    max_out_len = len(next_state)
                outputs.append(next_state)
            
        return states, actions, outputs, max_in_len, max_a_len, max_out_len
    
    def get_concat_samples(self, batch_idx, num_negative=10, truncate_num=0):
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
            
            wt_question = state + [self.SEP_TOKEN] + wt_action
            if truncate_num > 0:
                wt_question = wt_question[:truncate_num]
            if len(wt_question) > max_in_len:
                max_in_len = len(wt_question)
            concat_inputs.append(wt_question)
            
            wt_next_state = wt_next_state
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
                
                question = state + [self.SEP_TOKEN] + action
                if truncate_num > 0:
                    question = question[:truncate_num]
                if len(question) > max_in_len:
                    max_in_len = len(question)
                concat_inputs.append(question)

                next_state = next_state
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
                
                question = state + [self.SEP_TOKEN] + wt_action
                if truncate_num > 0:
                    question = question[:truncate_num]
                if len(question) > max_in_len:
                    max_in_len = len(question)
                concat_inputs.append(question)

                if truncate_num > 0:
                    next_state = next_state[:truncate_num]
                if len(next_state) > max_out_len:
                    max_out_len = len(next_state)
                outputs.append(next_state)
            
        return concat_inputs, outputs, max_in_len, max_out_len

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
            
            wt_question = state
            if truncate_num > 0:
                wt_question = wt_question[:truncate_num]
            if len(wt_question) > max_in_len:
                max_in_len = len(wt_question)
            states.append(wt_question)
            
            if len(wt_action) > max_a_len:
                max_a_len = len(wt_action)
            actions.append(wt_action)
            
            wt_next_state = wt_next_state
            if truncate_num > 0:
                wt_next_state = wt_next_state[:truncate_num]
            if len(wt_next_state) > max_out_len:
                max_out_len = len(wt_next_state)
            outputs.append(wt_next_state)
            
            # sample other actions under the state                
            for neg_idx in range(len(pair_dict_["actions"])):
                action = pair_dict_['actions'][neg_idx]
                next_state = pair_dict_['next_states'][neg_idx]
                
                question = state
                if truncate_num > 0:
                    question = question[:truncate_num]
                if len(question) > max_in_len:
                    max_in_len = len(question)
                states.append(question)
                
                if len(action) > max_a_len:
                    max_a_len = len(action)
                actions.append(action)

                if truncate_num > 0:
                    next_state = next_state[:truncate_num]
                if len(next_state) > max_out_len:
                    max_out_len = len(next_state)
                outputs.append(next_state)
            
        return states, actions, outputs, max_in_len, max_a_len, max_out_len
    
    
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
            
            wt_question = state + [self.SEP_TOKEN] + wt_action
            if truncate_num > 0:
                wt_question = wt_question[:truncate_num]
            if len(wt_question) > max_in_len:
                max_in_len = len(wt_question)
            concat_inputs.append(wt_question)
            
            wt_next_state = wt_next_state
            if truncate_num > 0:
                wt_next_state = wt_next_state[:truncate_num]
            if len(wt_next_state) > max_out_len:
                max_out_len = len(wt_next_state)
            outputs.append(wt_next_state)
            
            # sample other actions under the state                
            for neg_idx in range(len(pair_dict_["actions"])):
                action = pair_dict_['actions'][neg_idx]
                next_state = pair_dict_['next_states'][neg_idx]
                
                question = state + [self.SEP_TOKEN] + action
                if truncate_num > 0:
                    question = question[:truncate_num]
                if len(question) > max_in_len:
                    max_in_len = len(question)
                concat_inputs.append(question)

                if truncate_num > 0:
                    next_state = next_state[:truncate_num]
                if len(next_state) > max_out_len:
                    max_out_len = len(next_state)
                outputs.append(next_state)
            
        return concat_inputs, outputs, max_in_len, max_out_len
    
    
    def check_eval_triples(self, vocab):
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

            num_cand = 0
            obs_dict = {}
            for neg_idx in range(len(pair_dict_["actions"])):
                action = pair_dict_['actions'][neg_idx]
                next_state = pair_dict_['next_states'][neg_idx]

                next_state_words = [vocab[wid] for wid in next_state]
                next_state_str = ' '.join(next_state_words)
                
                if next_state_str not in obs_dict:
                    obs_dict[next_state_str] = 1
                    num_cand += 1
                    
            if num_cand > max_cand:
                max_cand = num_cand
            total_cand += num_cand

            total_state += 1
        print('max number of cand:', max_cand)
        print('avg number of cand: {}/{}={}'.format(total_cand, total_state, total_cand/total_state))
            
    def print_info(self):
        print('Number of walkthrough tuples: {}'.format(self.num_positive))
        print('Number of tuples: {}'.format(self.num_tuples))
        print('Number of unique actions: {}'.format(len(self.action2tuples)))


# In[ ]:


class StateAction2StateDataset(StateState2ActionDataset):
    
    def __init__(self, data_dir, rom_dir, game2rom, train_games=None, dev_games=None, setting='same_games',
                 num_negative=20, truncate_num=300, freq_threshold=2):     
        super(StateAction2StateDataset, self).__init__(data_dir, rom_dir, game2rom,
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
        datasets['train'] = ForwardPredictionSet()
        datasets['dev'] = ForwardPredictionSet()
        datasets['test'] = ForwardPredictionSet()
        
        avg_cand_rouge_l = 0
        total_cand = 0
        
        avg_wt_rouge_l = 0
        total_wt_cand = 0
        
        for game_name in games:
#             rom_path = "../roms/jericho-game-suite/{}.z5".format(game_name)
            print('# LOADING game data {} ...'.format(game_name))
    
            num_unmatched_wt_action = 0
            
            f = open(os.path.join(self.data_dir, '{}.sas.wt_traj.tok'.format(game_name)), "r")
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
        datasets['train'] = ForwardPredictionSet()
        datasets['dev'] = ForwardPredictionSet()
        datasets['test'] = ForwardPredictionSet()
        
        avg_cand_rouge_l = 0
        total_cand = 0
        
        avg_wt_rouge_l = 0
        total_wt_cand = 0
        
        for game_name in games + dev_games:
#             rom_path = "../roms/jericho-game-suite/{}.z5".format(game_name)
            print('# LOADING game data {} ...'.format(game_name))
    
            num_unmatched_wt_action = 0
            
            f = open(os.path.join(self.data_dir, '{}.sas.wt_traj.tok'.format(game_name)), "r")
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
                        
                    if game_name in dev_games:
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

                if game_name not in dev_games:
#                     datasets['train'].add_one(state, next_states, actions, wt_next_state, wt_action)
                    datasets['train'].add_one(state, next_states_, actions_, wt_next_state, wt_action)
                else:
#                     rouge_score = calc_score([wt_next_state], [state])
#                     avg_wt_rouge_l += rouge_score
#                     total_wt_cand += 1

#                     actions__ = []
#                     next_states__ = []

#                     rouge_scores = []

#                     obs_dict = {wt_next_state:1}
#                     for neg_idx in range(len(actions)):
#                         action = actions[neg_idx]
#                         next_state = next_states[neg_idx]

#                         if action.startswith('drop') and len(action.split(' ')) == 2:
#                             continue

#                         if next_state not in obs_dict:
#                             rouge_score = calc_score([next_state], [state])                            

#                             obs_dict[next_state] = 1
#                             actions__.append(action)
#                             next_states__.append(next_state)
#                             rouge_scores.append((len(rouge_scores), rouge_score))

#                     sorted_rouge_scores = sorted(rouge_scores, key = lambda x:x[1])
#                     actions_ = []
#                     next_states_ = []
#                     for (neg_idx, _) in sorted_rouge_scores:
#                         action = actions__[neg_idx]
#                         next_state = next_states__[neg_idx]

#                         actions_.append(action)
#                         next_states_.append(next_state)

#                         avg_cand_rouge_l += rouge_score
#                         total_cand += 1

#                         if len(actions_) == 15:
#                             break
                            
                    if idx / len(instances) < 0.5:
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
    
    def _numeralize_pairs(self, word_freq_dict, pairs):
        '''
        numeralize passages in training pair lists
        '''
        ret_pair_list = []
        for pair_dict_ in pairs:
            new_pair_dict_ = {}

            for k, v in pair_dict_.items():
                if k == 'state' or k == 'wt_next_state':
                    new_pair_dict_[k] = self._add_vocab_from_sentence(word_freq_dict, v)
                elif k == 'wt_action':
                    new_pair_dict_[k] = self._add_vocab_from_sentence(word_freq_dict, v)
                    new_pair_dict_['wt_action_str'] = v
                elif k == 'next_states':
                    new_pair_dict_[k] = []
                    for seq in v:
                        new_pair_dict_[k].append(self._add_vocab_from_sentence(word_freq_dict, seq))
                elif k == 'actions':
                    new_pair_dict_[k] = []
                    for seq in v:
                        new_pair_dict_[k].append(self._add_vocab_from_sentence(word_freq_dict, seq))

            ret_pair_list.append(new_pair_dict_)
        return ret_pair_list

        
    def _get_word_freq(self, data_sets_):
        """
        Building word frequency dictionary and filter the vocabulary
        """

        word_freq_dict = {}

        for data_id, data_set in data_sets_.items():
            for pair_dict in data_set.get_pairs():
                for k, v in pair_dict.items():
                    if k == 'state' or k == 'wt_next_state':
                        self._add_freq_from_sentence(word_freq_dict, v)
                    elif k == 'wt_action':
                        self._add_freq_from_sentence(word_freq_dict, v)
                    elif k == 'next_states':
                        for seq in v:
                            self._add_freq_from_sentence(word_freq_dict, seq)
                    elif k == 'actions':
                        for seq in v:
                            self._add_freq_from_sentence(word_freq_dict, seq)

        print('size of the raw vocabulary:', len(word_freq_dict))
        return word_freq_dict
    
        
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
        concat_inputs, outputs, max_in_len, max_out_len = data_set.get_concat_samples(batch_idx, 
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
    
#     game_task_data = StateState2ActionDataset(data_dir, rom_dir=rom_dir, game2rom=game2rom,
#                                               train_games=games, dev_games=games,
#                                               setting='same_games')
# #     game_task_data = StateState2ActionDataset(data_dir, train_games, dev_games)
    
#     x1, x2, positives, candidates, max_x1_len, max_x2_len, max_a_len = game_task_data.data_sets['train'].get_samples_from_one_list([0,1])

#     print(x1)
#     print(x2)
#     print(positives)
#     print(candidates)

#     x1_mat, x2_mat, a_mat, y_vec, x1_mask, x2_mask, a_mask = game_task_data.get_batch('train', [0,1])

#     print(x1_mat)
#     print(a_mat)

#     game_task_data.display_sentence(x1[0])
#     game_task_data.display_sentence(x1_mat[0])

#     game_task_data.display_sentence(x2[0])
#     game_task_data.display_sentence(x2_mat[0])
#     game_task_data.display_sentence(x2_mat[1])

# #     print(positives)
# #     print(candidates)

# #     game_task_data.display_sentence(positives[0])
# #     game_task_data.display_sentence(positives[1])
# #     game_task_data.display_sentence(candidates[0][3])
# #     game_task_data.display_sentence(a_mat[0][0])
# #     game_task_data.display_sentence(a_mat[1][0])
# #     game_task_data.display_sentence(a_mat[0][4])

#     x_mat, a_mat, y_vec, x_mask, a_mask = game_task_data.get_batch_concat('train', [0,1])

#     game_task_data.display_sentence(x_mat[0])
    
#     print(x_mat.shape)
#     print(a_mat.shape)
#     print(x_mask.shape)
#     print(a_mask.shape)
#     print(y_vec.shape)

    data_dir = "/dccstor/yum-worldmodel/shared_folder_2080/if_games/data/ssa_data/zork_universe_sup/"
    
    
    games = ['zork1', 'zork3']
    dev_games = ['zork3']
#     games = ['zork1', 'zork3', 'enchanter', 'spellbrkr', 'sorcerer']
    games = ['zork1', 'zork3', 'enchanter', 'sorcerer']
    train_games = ['zork1', 'enchanter', 'sorcerer']
    dev_games = ['zork3']
#     games = ['spellbrkr']
#     games = ['zork1']
    
    rom_dir = '../roms/jericho-game-suite/'
    game2rom = find_game_roms(games, rom_dir)
    print(game2rom)
    
    pretrain_path = '/dccstor/gaot1/MultiHopReason/comprehension_tasks/narrativeqa/passage_ranker/bert-base-uncased/'
    
#     game_task_data = StateAction2StateDataset(data_dir, rom_dir=rom_dir, game2rom=game2rom,
#                                               train_games=games, dev_games=games,
#                                               setting='same_games')

    game_task_data = StateAction2StateDataset(data_dir, rom_dir=rom_dir, game2rom=game2rom,
                                              train_games=train_games, dev_games=dev_games,
                                              setting='transfer')

    game_task_data.data_sets['dev'].check_eval_triples(game_task_data.idx_2_word)
    game_task_data.data_sets['test'].check_eval_triples(game_task_data.idx_2_word)

    
    i_mat, o_mat, y_vec, i_mask, o_mask = game_task_data.get_eval_batch_concat('train', [2])

    for i in range(i_mat.shape[0]):
        game_task_data.display_sentence(i_mat[i])
        game_task_data.display_sentence(o_mat[i])
    #     print(concat_inputs[i])
    #     print(outputs[i])
        print('')
        
    print(y_vec)
    print(i_mat.shape)
    
    concat_inputs, outputs, max_in_len, max_out_len = game_task_data.data_sets['train'].get_eval_concat_samples([2])

    print(len(concat_inputs))

    for i in range(len(concat_inputs)):
        game_task_data.display_sentence(concat_inputs[i])
        game_task_data.display_sentence(outputs[i])
    #     print(concat_inputs[i])
    #     print(outputs[i])
        print('')

    i_mat, a_mat, o_mat, y_vec, i_mask, a_mask, o_mask = game_task_data.get_eval_batch_triple('train', [2])

    for i in range(i_mat.shape[0]):
        game_task_data.display_sentence(i_mat[i])
        game_task_data.display_sentence(a_mat[i])
        game_task_data.display_sentence(o_mat[i])
    #     print(concat_inputs[i])
    #     print(outputs[i])
        print('')


# In[ ]:


#     concat_inputs, outputs, max_in_len, max_out_len = game_task_data.data_sets['test'].get_eval_concat_samples([111])

#     print(len(concat_inputs))

#     for i in range(len(concat_inputs)):
#         game_task_data.display_sentence(concat_inputs[i])
#         game_task_data.display_sentence(outputs[i])

# states, actions, outputs, max_in_len, max_a_len, max_out_len = game_task_data.data_sets['train'].get_eval_triple_samples([2])

# print(len(states))

# for i in range(len(states)):
#     game_task_data.display_sentence(states[i])
#     game_task_data.display_sentence(actions[i])
#     game_task_data.display_sentence(outputs[i])
# #     print(concat_inputs[i])
# #     print(outputs[i])
#     print('')


# In[ ]:


# # set_id = 'train'
# # data_set = game_task_data.data_sets[set_id]
# # batch_idx = np.random.randint(0, data_set.size(), size=40)
# # print(batch_idx)

# # x_mat, a_mat, y_vec, x_mask, a_mask = game_task_data.get_train_batch(40, inst_format='concat')
# # print(x_mat.shape)
# # print(a_mat.shape)
# # print(x_mask.shape)
# # print(a_mask.shape)
# # print(y_vec.shape)

# x_mat, a_mat, y_list, x_mask, a_mask = game_task_data.get_eval_batch_concat('train', 1)
# print(x_mat.shape)
# print(a_mat.shape)
# print(x_mask.shape)
# print(a_mask.shape)
# print(len(y_list))

# game_task_data.display_sentence(x_mat[0])
# for i in range(a_mat.shape[1]):
#     game_task_data.display_sentence(a_mat[0][i])
# print(y_list)

# x1_mat, x2_mat, a_mat, y_list, x1_mask, x2_mask, a_mask = game_task_data.get_eval_batch('train', 1)
# print(x1_mat.shape)
# print(a_mat.shape)
# print(x_mask.shape)
# print(a_mask.shape)
# print(len(y_list))

# game_task_data.display_sentence(x1_mat[0])
# game_task_data.display_sentence(x2_mat[0])
# for i in range(a_mat.shape[1]):
#     game_task_data.display_sentence(a_mat[0][i])
# print(y_list)


# In[ ]:



# rom_path = "../roms/jericho-game-suite/zork1.z5" # "../roms/jericho-game-suite/zork1.z5"

# bindings = load_bindings(rom_path)

# print(bindings['grammar'])


# In[ ]:


# act_par = TemplateActionParser(bindings)
# # print(act_par.templates_alias_dict)
# # print(act_par.verb_to_templates)

# f = open(os.path.join('.', 'zork1.ssa.wt_traj.txt'), "r")
# instances = f.readlines()

# def _process_instance(instance):
#     new_inst = json.loads(instance)
#     info = new_inst['observations'].split('|')
#     new_inst['observations'] = {'obs':' | '.join(info[:3]), 'action':info[3]}
#     return new_inst
            
# instances = [_process_instance(instance) for instance in instances]


# In[ ]:


# # print(act_par.templates[227])
# # print(act_par.templates[173])

# def _preprocess_action(action):
#     action = action.lower()
    
#     if action == 'n':
#         action = 'north'
#     elif action == 's':
#         action = 'south'
#     elif action == 'e':
#         action = 'east'
#     elif action == 'w':
#         action = 'west'
#     elif action == 'se':
#         action = 'southeast'
#     elif action == 'sw':
#         action = 'southwest'
#     elif action == 'ne':
#         action = 'northeast'
#     elif action == 'nw':
#         action = 'northwest'
#     elif action == 'u':
#         action = 'up'
#     elif action == 'd':
#         action = 'down'
#     return action

# 'north/south/west/east/northwest/southwest/northeast/southeast/up/down/enter/exit/take all'.split('/')

# for idx, instance in enumerate(instances):
    
# #     print(instance['valid_actions'])
#     action = _preprocess_action(instance['observations']['action'])
# #     print(action)
#     template = act_par.parse_action(action)
#     if template is None:
#         print('unmatched action: {}'.format(action))
#     elif template[0] not in act_par.template2template:
#         if template[0] not in act_par.add_template2template:
#             print('cannot find root: {}'.format(action))
#         else:
#             pass
#     else:
#         pass
    
# #     for a_list in instance['valid_actions']:
# #         for action in a_list:
# #             template_id = action['t']
# # #             print(act_par.templates[template_id])
            
# #             template = act_par.parse_action(action['a'])
# #             if template is None:
# #                 print('unmatched action: {}'.format(action['a']))
# #             print('{}: {}'.format(action['a'],template))


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


# act_str = 'apply elephant to fridge'
# act_str = 'fix/glue/patch/plug/repair OBJ with OBJ'
# act_str = 'glue elephant with fridge'
# template = act_par.parse_action(act_str)
# print('{}: {}'.format(act_str, template))
# print('root template: {}'.format(act_par.template2template[template[0]]))
# print('recovered action: {}'.format(_recover_root_template_action(template)))

# act_str = 'get out of boat'
# template = act_par.parse_action(act_str)
# print('{}: {}'.format(act_str, template))
# print('root template: {}'.format(act_par.template2template[template[0]]))
# print('recovered action: {}'.format(_recover_root_template_action(template)))

# act_str = 'kill thief with nasty knife'
# template = act_par.parse_action(act_str)
# print('{}: {}'.format(act_str, template))
# print('root template: {}'.format(act_par.template2template[template[0]]))
# print('recovered action: {}'.format(_recover_root_template_action(template)))

# act_str = 'drop rusty knife'
# template = act_par.parse_action(act_str)
# print('{}: {}'.format(act_str, template))
# print('root template: {}'.format(act_par.template2template[template[0]]))
# print('recovered action: {}'.format(_recover_root_template_action(template)))


# In[ ]:


# def _recover_root_template_action(template):
#     root_template = act_par.template2template[template[0]]
#     t_tokens = root_template.split()
#     count = 1
#     for tid, t_token in enumerate(t_tokens):
#         if t_token == 'OBJ':
#             t_tokens[tid] = template[count]
#             count += 1
            
#     return ' '.join(t_tokens)


# act_str = 'light candles with match'
# template = act_par.parse_action(act_str)
# print('{}: {}'.format(act_str, template))
# print('root template: {}'.format(act_par.template2template[template[0]]))
# print('recovered action: {}'.format(_recover_root_template_action(template)))


# act_str = 'exting lamp'
# template = act_par.parse_action(act_str)
# print('{}: {}'.format(act_str, template))
# print('root template: {}'.format(act_par.template2template[template[0]]))
# print('recovered action: {}'.format(_recover_root_template_action(template)))


# In[ ]:




