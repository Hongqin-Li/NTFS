import torch
import torch.nn as nn

class Dataset():
    def __init__(self, train_file, dev_file, test_file, word_to_idx):
        # word_to_idx/tag_to_idx: both are functions, whose input is a string and output an int

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.word_to_idx = word_to_idx

    def trainset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.train_file, batch_size=batch_size, drop_last=drop_last):
            yield batch

    def devset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.dev_file, batch_size=batch_size, drop_last=drop_last):
            yield batch

    def testset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.test_file, batch_size=batch_size, drop_last=drop_last):
            yield batch

    def sentence_to_tensor(self, s):
        # s: string 
        # return: 1-d long tensor of shape (seq_len)
        return torch.LongTensor([self.word_to_idx(w) for w in s])

    def pad_sequence(self, s):
        # TODO pad to 512?
        return nn.utils.rnn.pad_sequence(s, batch_first=True)

    def samples(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sen1,sen2,label = line.strip().split('\t')
                yield sen1+'[sep]'+sen2,label # use [seq] to seperate the sentence1 and sentence2

    def sample_batches(self, file_path, batch_size=1, drop_last=False):
        # drop_last: drop the last incomplete batch if True
        cnt = 0

        # Input
        sent_batch = [] # (batch_size, seq_len)
        # Target
        tag_batch = [] # (batch_size)
        for sent, tag in self.samples(file_path):
            sent_batch.append(self.sentence_to_tensor(sent))
            tag_batch.append(int(tag))

            cnt += 1

            if cnt >= batch_size:
                yield self.pad_sequence(sent_batch), torch.LongTensor(tag_batch)
                sent_batch, tag_batch = [], []
                cnt = 0
    
        if cnt>0 and not drop_last:
            yield self.pad_sequence(sent_batch), torch.LongTensor(tag_batch)

if __name__ == '__main__':
    # Usage
    train_file = 'train.txt'
    dev_file = 'dev.txt'
    test_file = 'test.txt'

    def word_to_idx(w):
        w2i = {'当': 1, '希': 2}
        return w2i.get(w, 0)

    dataset = Dataset(train_file=train_file, dev_file=dev_file, test_file=test_file, word_to_idx=word_to_idx)

    cnt = 0
    for sent_batch, tag_batch in dataset.trainset(batch_size=10, drop_last=False):
        cnt += sent_batch.shape[0]
    print (f'trainset: {cnt}')

    cnt = 0
    for sent_batch, tag_batch in dataset.devset(batch_size=10, drop_last=False):
        cnt += sent_batch.shape[0]
    print (f'devset: {cnt}')

    cnt = 0
    for sent_batch, tag_batch in dataset.testset(batch_size=10, drop_last=False):
        cnt += sent_batch.shape[0]
    print (f'testset: {cnt}')
