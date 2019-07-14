import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from collections import namedtuple

# Can be simply regarded as a object, called "Batch", having attributes "input" and "target"
Batch = namedtuple('Batch', 'input target idx_to_tag')

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

    def __init__(self, train_file=None, test_file=None, word_to_idx=None, max_seq_len=512, split=0.9, use_gpu=False):
        # word_to_idx: function, whose input is a string and output an int

        self.max_seq_len = max_seq_len
        self.use_gpu = use_gpu
        self.num_classes = 7

        self.train_file = train_file
        self.dev_file = 'dev'
        self.test_file = test_file

        self.word_to_idx = word_to_idx
        
        train_dev_samples = 0
        for _ in self.samples(train_file, begin_idx=0, end_idx=-1):
            train_dev_samples += 1

        self.num_train_samples = int(train_dev_samples * split)
        self.num_dev_samples = train_dev_samples - self.num_train_samples

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

    def words_to_tensor(self, ws):
        # ws: string or list
        # return: 1-d long tensor of shape (seq_len)
        return torch.LongTensor([self.word_to_idx(w) for w in ws])
    
    def tag_to_idx(self, tag):
        t2i = {'O':0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
        return t2i[tag]

    def idx_to_tag(self, i):
        i2t = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
        return i2t[i]

    def tags_to_tensor(self, ts):
        return torch.LongTensor([self.tag_to_idx(t) for t in ts])


    def samples(self, file_path, begin_idx, end_idx):
        # begin_idx and end_idx are used to split train_file into trainset and devset
        i = -1

        with open(file_path, 'r') as f:
            words = [] # list of char: (seq_len)
            tags = [] # list of tag(string): (seq_len)

            for line in f:

                if line.isspace():

                    i += 1
                    if i >= end_idx and end_idx > 0:
                        break

                    elif i >= begin_idx: 
                        yield words, tags

                    words, tags = [], []
                else:
                    ls = line.split()
                    words.append(ls[0])
                    tags.append(ls[1] if len(ls) > 1 else 'O') # Since the data is a bit of strange
   
    def sample_batches(self, file_path, batch_size=1, drop_last=False):
        # drop_last: drop the last incomplete batch if True
        cnt = 0

        # Input
        token_idxs_batch, token_type_idxs_batch, mask_batch = [], [], [] # (batch_size, seq_len)
        # Target
        tags_batch = [] # (batch_size, seq_len)

        if file_path == self.train_file:
            begin_idx, end_idx = 0, self.num_train_samples
        elif file_path == self.dev_file:
            begin_idx, end_idx = self.num_train_samples, -1
            file_path = self.train_file
        else:
            begin_idx, end_idx = 0, -1


        for words, tags in self.samples(file_path, begin_idx, end_idx):
            # all list-like

            token_idxs, token_type_idxs, mask = parse_sentence(words, self.word_to_idx, self.max_seq_len) # (seq_len)

            token_idxs = torch.LongTensor(token_idxs)
            token_type_idxs = torch.LongTensor(token_type_idxs)
            mask = torch.LongTensor(mask)

            if len(tags) > self.max_seq_len - 2: tags = tags[: self.max_seq_len-2]

            tags = self.tags_to_tensor(tags)

            if self.use_gpu:
                token_idxs, token_type_idxs, mask, tags = token_idxs.cuda(), token_type_idxs.cuda(), mask.cuda(), tags.cuda()

            token_idxs_batch.append(token_idxs)
            token_type_idxs_batch.append(token_type_idxs)
            mask_batch.append(mask)
            tags_batch.append(tags)

            cnt += 1

            if cnt >= batch_size:
                yield Batch(input=(pad_sequence(token_idxs_batch, batch_first=True),
                                   pad_sequence(token_type_idxs_batch, batch_first=True), 
                                   pad_sequence(mask_batch, batch_first=True)), 
                            target=pad_sequence(tags_batch, batch_first=True, padding_value=-1), 
                            idx_to_tag=self.idx_to_tag)

                token_idxs_batch, token_type_idxs_batch, mask_batch, tags_batch = [], [], [], []
                cnt = 0

        if cnt > 0 and not drop_last:
            yield Batch(input=(pad_sequence(token_idxs_batch, batch_first=True),
                               pad_sequence(token_type_idxs_batch, batch_first=True), 
                               pad_sequence(mask_batch, batch_first=True)), 
                        target=pad_sequence(tags_batch, batch_first=True, padding_value=-1), 
                        idx_to_tag=self.idx_to_tag)

if __name__ == '__main__': 
    # Usage
    train_file = 'msra_train_bio.txt'
    test_file = 'msra_test_bio.txt'

    def word_to_idx(w):
        w2i = {'当': 1, '希': 2}
        return w2i.get(w, 0)

    dataset = Dataset(train_file=train_file, test_file=test_file, word_to_idx=word_to_idx, split=0.9)

    print (f'trainset: {dataset.num_train_samples}')
    print (f'devset: {dataset.num_dev_samples}')
    print (f'testset: {dataset.num_test_samples}')

    cnt = 0
    for (token_idxs_batch, token_type_idxs_batch, mask_batch), tag_batch, idx_to_tag in dataset.trainset(batch_size=10, drop_last=False):
        # print (f'input_batch: {token_idxs_batch.shape, token_type_idxs_batch.shape, mask_batch.shape}, target_batch: {tag_batch.shape}')
        # print (token_idxs_batch)
        # input ()
        cnt += tag_batch.shape[0]
    print (f'trainset: {cnt}')
    
    cnt = 0
    for (token_idxs_batch, token_type_idxs_batch, mask_batch), tag_batch, idx_to_tag in dataset.devset(batch_size=10, drop_last=False):
        # print (f'input_batch: {token_idxs_batch.shape, token_type_idxs_batch.shape, mask_batch.shape}, target_batch: {tag_batch.shape}')
        # input ()
        cnt += tag_batch.shape[0]
    print (f'devset: {cnt}')

    cnt = 0
    for (token_idxs_batch, token_type_idxs_batch, mask_batch), tag_batch, idx_to_tag in dataset.testset(batch_size=10, drop_last=False):
        # print (f'input_batch: {token_idxs_batch.shape, token_type_idxs_batch.shape, mask_batch.shape}, target_batch: {tag_batch.shape}')
        # input ()
        cnt += tag_batch.shape[0]
    print (f'testset: {cnt}')
 
