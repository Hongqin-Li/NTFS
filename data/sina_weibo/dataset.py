import os
import random
import json
import torch
import torch.nn as nn

class Dataset(): 

    def __init__(self, raw_file, train_file, dev_file, test_file, word_to_idx):
        # word_to_idx/tag_to_idx: both are functions, whose input is a string and output an int

        self.raw_file = raw_file

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.word_to_idx = word_to_idx

        self.split_raw()
        
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


    def split_raw(self):
        num_dev_file = 10000
        num_test_file = 10000

        if os.path.exists(self.train_file) or os.path.exists(self.dev_file) or os.path.exists(self.test_file):
            print ('Find previous split files! No need to split again.')
            return

        print ('Splitting...', end='')
        with open(self.raw_file, 'r') as f:

            pos_lines = []
            neg_lines = []

            f.readline() # Omit header
            for line in f:
                assert line[0] == '0' or line[0] == '1'
                assert line[1] == ','
                assert '\t' not in line
                if line[0] == '1': pos_lines.append(line)
                else: neg_lines.append(line)
        
            random.shuffle(pos_lines)
            random.shuffle(neg_lines)

            dev_end = num_dev_file // 2
            test_end = (num_test_file + num_dev_file) // 2
    
            dev_lines = pos_lines[: dev_end] + neg_lines[: dev_end]
            test_lines = pos_lines[dev_end: test_end] + neg_lines[dev_end: test_end]
            train_lines = pos_lines[test_end:] + neg_lines[test_end:]

            with open(self.train_file, 'w') as f:
                f.write('label\ttext\n')
                for line in train_lines:
                    f.write(f'{line[0]}\t{line[2:]}')

            with open(self.dev_file, 'w') as f:
                f.write('label\ttext\n')
                for line in dev_lines:
                    f.write(f'{line[0]}\t{line[2:]}')

            with open(self.test_file, 'w') as f:
                f.write('label\ttext\n')
                for line in test_lines:
                    f.write(f'{line[0]}\t{line[2:]}')
            print ('Finish!')

    def sentence_to_tensor(self, s):
        # s: string or list
        # return: 1-d long tensor of shape (seq_len)
        return torch.LongTensor([self.word_to_idx(w) for w in s])

    def pad_sequence(self, s):
        # TODO pad to 512?
        return nn.utils.rnn.pad_sequence(s, batch_first=True)

    def samples(self, file_path):

        with open(file_path, 'r') as f:

            labels = f.readline().strip().split('\t') # Omit tsv header

            for line in f:
                tag, sent = line.strip().split('\t')
                # print (tag, sent)
                yield sent, tag
                
   
    def sample_batches(self, file_path, batch_size=1, drop_last=False):
        # drop_last: drop the last incomplete batch if True
        cnt = 0

        # Input
        sent_batch = [] # (batch_size, seq_len)
        # Target
        tag_batch = [] # (batch_size)

        for sent, tag in self.samples(file_path):
            # all string-like

            sent_batch.append(self.sentence_to_tensor(sent))
            tag_batch.append(int(tag))

            cnt += 1

            if cnt >= batch_size:
                yield self.pad_sequence(sent_batch), torch.LongTensor(tag_batch)
                sent_batch, tag_batch = [], []
                cnt = 0

        if cnt > 0 and not drop_last:
            yield self.pad_sequence(sent_batch), torch.LongTensor(tag_batch)




if __name__ == '__main__': 
    # Usage
    raw_file = 'weibo_senti_100k.csv'
    train_file = 'train.tsv'
    dev_file = 'dev.tsv'
    test_file = 'test.tsv'

    def word_to_idx(w):
        w2i = {'当': 1, '希': 2}
        return w2i.get(w, 0)

    dataset = Dataset(raw_file=raw_file, train_file=train_file, dev_file=dev_file, test_file=test_file, word_to_idx=word_to_idx)

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


