import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from collections import namedtuple

# Can be simply regarded as a object, called "Batch", having attributes "input" and "target"
Batch = namedtuple('Batch', 'input target idx_to_tag')

class Dataset(): 

    def __init__(self, train_file, dev_file, test_file, word_to_idx, use_gpu=False):
        # vocab: dict-like, map word to idx, e.g. vocab['a'] = 1
        self.use_gpu = use_gpu
        self.num_classes = 7

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.word_to_idx = word_to_idx

        self.num_train_samples = 0
        for batch in self.trainset(batch_size=1000):
            self.num_train_samples += batch[0].shape[0]

        self.num_dev_samples = 0
        for batch in self.devset(batch_size=1000):
            self.num_dev_samples += batch[0].shape[0]

        self.num_test_samples = 0
        for batch in self.testset(batch_size=1000):
            self.num_test_samples += batch[0].shape[0]


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

    def samples(self, file_path):

        with open(file_path, 'r') as f:

            words = [] # list of char: (seq_len)
            tags = [] # list of string: (seq_len)

            for line in f:
                if line.isspace():

                    words = self.words_to_tensor(words)
                    tags = self.tags_to_tensor(tags)

                    if self.use_gpu:
                        yield words.cuda(), tags.cuda()
                    else:
                        yield words, tags

                    words, tags = [], []
                else:
                    w, t = line.split()
                    words.append(w)
                    tags.append(t)
            

   
    def sample_batches(self, file_path, batch_size=1, drop_last=False):
        # drop_last: drop the last incomplete batch if True
        cnt = 0

        # Input
        words_batch = [] # (batch_size, seq_len)
        # Target
        tags_batch = [] # (batch_size, seq_len)

        for words, tags in self.samples(file_path):
            # all tensor-like
            words_batch.append(words)
            tags_batch.append(tags)

            cnt += 1

            if cnt >= batch_size:
                yield Batch(input=pad_sequence(words_batch, batch_first=True, padding_value=0), 
                            target=pad_sequence(tags_batch, batch_first=True, padding_value=-1), 
                            idx_to_tag=self.idx_to_tag)
                words_batch, tags_batch = [], []
                cnt = 0

        if cnt > 0 and not drop_last:
            yield Batch(input=pad_sequence(words_batch, batch_first=True, padding_value=0), 
                        target=pad_sequence(tags_batch, batch_first=True, padding_value=-1), 
                        idx_to_tag=self.idx_to_tag)


if __name__ == '__main__': 
    # Usage
    train_file = 'train.txt'
    dev_file = 'dev.txt'
    test_file = 'test.txt'

    def word_to_idx(w):
        w2i = {'当': 1, '希': 2}
        return w2i.get(w, 0)


    dataset = Dataset(train_file=train_file, dev_file=dev_file, test_file=test_file, word_to_idx=word_to_idx)

    print (f'trainset: {dataset.num_train_samples}')
    print (f'devset: {dataset.num_dev_samples}')
    print (f'testset: {dataset.num_test_samples}')

    cnt = 0
    for words_batch, tags_batch, idx_to_tag in dataset.trainset(batch_size=10, drop_last=False):
        # words_batch/tags_batch: (batch_size, seq_len)
        # print (f'words_batch: {words_batch}')
        # print (f'tags_batch: {tags_batch}')
        # print (f'words_batch: {words_batch.shape}, tags_batch: {tags_batch.shape}')
        # print ([[idx_to_tag(i) for i in tags] for tags in tags_batch.tolist()])
        # input ()
        cnt += words_batch.shape[0]
    print (f'trainset: {cnt}')
    
    cnt = 0
    for words_batch, tags_batch, idx_to_tag in dataset.devset(batch_size=10, drop_last=False):
        # words_batch/tags_batch: (batch_size, seq_len)
        # print (f'words_batch: {words_batch}')
        # print (f'tags_batch: {tags_batch}')
        # print (f'words_batch: {words_batch.shape}, tags_batch: {tags_batch.shape}')
        # input ()
        cnt += words_batch.shape[0]
    print (f'devset: {cnt}')

    cnt = 0
    for words_batch, tags_batch, idx_to_tag in dataset.testset(batch_size=10, drop_last=False):
        # words_batch/tags_batch: (batch_size, seq_len)
        # print (f'words_batch: {words_batch}')
        # print (f'tags_batch: {tags_batch}')
        # print (f'words_batch: {words_batch.shape}, tags_batch: {tags_batch.shape}')
        # input ()
        cnt += words_batch.shape[0]
    print (f'testset: {cnt}')
