import os
import random
import json
import torch
import torch.nn as nn

from collections import namedtuple

# Can be simply regarded as a object, called "Batch", having attributes "input" and "target"
Batch = namedtuple('Batch', 'input target')

# bert-like processing
def parse_sentence(sent, word_to_idx, max_seq_len):
    token_idxs, token_type_idxs, mask = [word_to_idx('[CLS]')], [0], [1] # (seq_len)

    length = len(sent)
    if length > max_seq_len - 2:
        # print ('Warning: exceed max_seq_len')
        length = max_seq_len - 2
        sent = sent[:length]

    token_idxs += [word_to_idx(w) for w in sent] + [word_to_idx('[SEP]')]
    token_type_idxs += [0] * (length + 1) # +1 for [SEP] following the sentence
    mask += [1] * (length + 1) # +1 for [SEP]

    assert len(token_idxs) == len(token_type_idxs) and len(token_idxs) == len(mask) and len(mask) <= max_seq_len

    return token_idxs, token_type_idxs, mask

class Dataset(): 

    def __init__(self, train_file, dev_file, test_file, word_to_idx, tag_to_idx, max_seq_len=512, use_gpu=False):
        # word_to_idx: function, whose input is a string and output an int

        self.max_seq_len = max_seq_len
        self.use_gpu = use_gpu

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx

        self.num_train_samples = 0
        for batch in self.trainset(batch_size=1000):
            self.num_train_samples += batch.input[0].shape[0]

        self.num_dev_samples = 0
        for batch in self.devset(batch_size=1000):
            self.num_dev_samples += batch.input[0].shape[0]

        self.num_test_samples = 0
        for batch in self.testset(batch_size=1000):
            self.num_test_samples += batch.input[0].shape[0]

    def trainset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.train_file, batch_size=batch_size, drop_last=drop_last):
            yield batch
            
    def devset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.dev_file, batch_size=batch_size, drop_last=drop_last):
            yield batch

    def testset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.test_file, batch_size=batch_size, drop_last=drop_last):
            yield batch

    def pad_sequence(self, s):
        return nn.utils.rnn.pad_sequence(s, batch_first=True)

    def samples(self, file_path):

        with open(file_path, 'r') as f:
            for line in f:
                tag, sent = line.strip().split('\t')
                yield sent, tag
                
   
    def sample_batches(self, file_path, batch_size=1, drop_last=False):
        # drop_last: drop the last incomplete batch if True
        cnt = 0

        # Input
        token_idxs_batch, token_type_idxs_batch, mask_batch = [], [], [] # (batch_size, seq_len)
        # Target
        tag_batch = [] # (batch_size)

        for sent, tag in self.samples(file_path):
            # all string-like

            token_idxs, token_type_idxs, mask = parse_sentence(sent, self.word_to_idx, self.max_seq_len)

            token_idxs = torch.LongTensor(token_idxs)
            token_type_idxs = torch.LongTensor(token_type_idxs)
            mask = torch.LongTensor(mask)
            tag = torch.LongTensor([self.tag_to_idx(tag)])

            if self.use_gpu:
                token_idxs, token_type_idxs, mask, tag = token_idxs.cuda(), token_type_idxs.cuda(), mask.cuda(), tag.cuda()

            token_idxs_batch.append(token_idxs)
            token_type_idxs_batch.append(token_type_idxs)
            mask_batch.append(mask)
            tag_batch.append(tag)

            cnt += 1

            if cnt >= batch_size:

                yield Batch(input=(self.pad_sequence(token_idxs_batch), 
                                   self.pad_sequence(token_type_idxs_batch), 
                                   self.pad_sequence(mask_batch)), 
                            target=torch.cat(tag_batch))

                token_idxs_batch, token_type_idxs_batch, mask_batch, tag_batch = [], [], [], []
                cnt = 0

        if cnt > 0 and not drop_last:
            yield Batch(input=(self.pad_sequence(token_idxs_batch), 
                               self.pad_sequence(token_type_idxs_batch), 
                               self.pad_sequence(mask_batch)), 
                        target=torch.cat(tag_batch))

if __name__ == '__main__': 
    # Usage
    train_file = 'cnews/cnews.train.txt'
    dev_file = 'cnews/cnews.val.txt'
    test_file = 'cnews/cnews.test.txt'

    def word_to_idx(w):
        w2i = {'当': 1, '希': 2}
        return w2i.get(w, 0)

    def tag_to_idx(t):
        t2i = {'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
        return t2i[t]

    dataset = Dataset(train_file=train_file, dev_file=dev_file, test_file=test_file, word_to_idx=word_to_idx, tag_to_idx=tag_to_idx)

    print (f'trainset: {dataset.num_train_samples}')
    print (f'devset: {dataset.num_dev_samples}')
    print (f'testset: {dataset.num_test_samples}')

    cnt = 0
    for (token_idxs_batch, token_type_idxs_batch, mask_batch), tag_batch in dataset.trainset(batch_size=10, drop_last=False):
        # print (f'input_batch: {token_idxs_batch.shape, token_type_idxs_batch.shape, mask_batch.shape}, target_batch: {tag_batch.shape}')
        # print (token_idxs_batch)
        # input ()
        cnt += tag_batch.shape[0]
    print (f'trainset: {cnt}')
    
    cnt = 0
    for (token_idxs_batch, token_type_idxs_batch, mask_batch), tag_batch in dataset.devset(batch_size=10, drop_last=False):
        # print (f'input_batch: {token_idxs_batch.shape, token_type_idxs_batch.shape, mask_batch.shape}, target_batch: {tag_batch.shape}')
        # input ()
        cnt += tag_batch.shape[0]
    print (f'devset: {cnt}')

    cnt = 0
    for (token_idxs_batch, token_type_idxs_batch, mask_batch), tag_batch in dataset.testset(batch_size=10, drop_last=False):
        # print (f'input_batch: {token_idxs_batch.shape, token_type_idxs_batch.shape, mask_batch.shape}, target_batch: {tag_batch.shape}')
        # input ()
        cnt += tag_batch.shape[0]
    print (f'testset: {cnt}')
    

