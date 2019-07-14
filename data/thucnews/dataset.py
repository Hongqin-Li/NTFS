import os
import random
import json
import torch
import torch.nn as nn

from collections import namedtuple

# Can be simply regarded as a object, called "Batch", having attributes "input" and "target"
Batch = namedtuple('Batch', 'input target')

class Dataset(): 

    def __init__(self, raw_train_file, train_file, dev_file, test_file, word_to_idx, use_gpu=False):
        # word_to_idx/tag_to_idx: both are functions, whose input is a string and output an int
        self.num_classes = 10
        self.use_gpu = use_gpu

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        # Shuffle trainset
        self.shuffle_trainset(raw_train_file, train_file)

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

    def shuffle_trainset(self, raw_file, out_file):

        if os.path.exists(out_file):
            print ('Trainset has been shuffled, do not shuffle.')
            return 

        lines = []
        with open(raw_file, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            with open(out_file, 'w') as f:
                for line in lines:
                    f.write(line)
 

    def trainset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.train_file, batch_size=batch_size, drop_last=drop_last):
            yield batch
            
    def devset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.dev_file, batch_size=batch_size, drop_last=drop_last):
            yield batch

    def testset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.test_file, batch_size=batch_size, drop_last=drop_last):
            yield batch

    def tag_to_idx(self, t):
        t2i = {'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
        return t2i[t]

    def sentence_to_tensor(self, s):
        # s: string or list
        # return: 1-d long tensor of shape (seq_len)
        return torch.LongTensor([self.word_to_idx(w) for w in s])

    def pad_sequence(self, s):
        # TODO pad to 512?
        return nn.utils.rnn.pad_sequence(s, batch_first=True)

    # TODO preprocess string
    def samples(self, file_path):

        with open(file_path, 'r') as f:
            for line in f:
                tag, sent = line.strip().split('\t')
                # print (tag, sent)
                # input ()

                sent = self.sentence_to_tensor(sent)
                tag = torch.LongTensor([self.tag_to_idx(tag)])

                if self.use_gpu: 
                    yield sent.cuda(), tag.cuda()
                else: 
                    yield sent, tag
                
   
    def sample_batches(self, file_path, batch_size=1, drop_last=False):
        # drop_last: drop the last incomplete batch if True
        cnt = 0

        # Input and target
        sent_batch, tag_batch = [], [] # (batch_size, seq_len), (batch_size)

        for sent, tag in self.samples(file_path):
            # all tensor-like

            sent_batch.append(sent)
            tag_batch.append(tag)

            cnt += 1
            if cnt >= batch_size:

                yield Batch(input=self.pad_sequence(sent_batch), target=torch.cat(tag_batch))
                sent_batch, tag_batch = [], []
                cnt = 0

        if cnt > 0 and not drop_last:
            yield Batch(input=self.pad_sequence(sent_batch), target=torch.cat(tag_batch))




if __name__ == '__main__': 
    # Usage
    raw_train_file = 'cnews/cnews.train.txt'
    train_file = 'cnews/cnews.train.shuffled.txt'
    dev_file = 'cnews/cnews.val.txt'
    test_file = 'cnews/cnews.test.txt'

    def word_to_idx(w):
        w2i = {'当': 1, '希': 2}
        return w2i.get(w, 0)

    dataset = Dataset(raw_train_file=raw_train_file, train_file=train_file, dev_file=dev_file, test_file=test_file, word_to_idx=word_to_idx)

    print (f'trainset: {dataset.num_train_samples}')
    print (f'devset: {dataset.num_dev_samples}')
    print (f'testset: {dataset.num_test_samples}')

    cnt = 0
    for sent_batch, tag_batch in dataset.trainset(batch_size=10, drop_last=False):
        # print (f'sent_batch: {sent_batch.shape}, tag_batch: {tag_batch.shape}')
        # input ()
        cnt += sent_batch.shape[0]
    print (f'trainset: {cnt}')
    
    cnt = 0
    for sent_batch, tag_batch in dataset.devset(batch_size=10, drop_last=False):
        # print (f'sent_batch: {sent_batch.shape}, tag_batch: {tag_batch.shape}')
        # input ()
        cnt += sent_batch.shape[0]
    print (f'devset: {cnt}')

    cnt = 0
    for sent_batch, tag_batch in dataset.testset(batch_size=10, drop_last=False):
        # print (f'sent_batch: {sent_batch.shape}, tag_batch: {tag_batch.shape}')
        # input ()
        cnt += sent_batch.shape[0]
    print (f'testset: {cnt}')

