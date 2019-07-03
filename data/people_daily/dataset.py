import json
import torch
import torch.nn as nn

class Dataset(): 

    def __init__(self, train_file, dev_file, test_file, word_to_idx, tag_to_idx):
        # vocab: dict-like, map word to idx, e.g. vocab['a'] = 1

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx

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
    
    def tags_to_tensor(self, ts):
        return torch.LongTensor([self.tag_to_idx(t) for t in ts])

    def pad_sequence(self, s):
        # TODO pad to 512?
        return nn.utils.rnn.pad_sequence(s, batch_first=True)


    def samples(self, file_path):

        with open(file_path, 'r') as f:

            words = [] # list of char: (seq_len)
            tags = [] # list of string: (seq_len)

            for line in f:
                if line.isspace():
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
            words_batch.append(self.words_to_tensor(words))
            tags_batch.append(self.tags_to_tensor(tags))

            cnt += 1

            if cnt >= batch_size:
                yield self.pad_sequence(words_batch), self.pad_sequence(tags_batch)
                words_batch, tags_batch = [], []
                cnt = 0

        if cnt > 0 and not drop_last:
            yield self.pad_sequence(words_batch), self.pad_sequence(tags_batch)




if __name__ == '__main__': 
    # Usage
    train_file = 'train.txt'
    dev_file = 'dev.txt'
    test_file = 'test.txt'

    def word_to_idx(w):
        w2i = {'当': 1, '希': 2}
        return w2i.get(w, 0)

    def tag_to_idx(tag):
        t2i = {'O':0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
        return t2i[tag]

    dataset = Dataset(train_file=train_file, dev_file=dev_file, test_file=test_file, word_to_idx=word_to_idx, tag_to_idx=tag_to_idx)

    cnt = 0
    for words_batch, tags_batch in dataset.trainset(batch_size=10, drop_last=False):
        # words_batch/tags_batch: (batch_size, seq_len)
        # print (f'words_batch: {words_batch}')
        # print (f'tags_batch: {tags_batch}')
        # print (f'words_batch: {words_batch.shape}, tags_batch: {tags_batch.shape}')
        # input ()
        cnt += words_batch.shape[0]
    print (f'trainset: {cnt}')
    
    cnt = 0
    for words_batch, tags_batch in dataset.devset(batch_size=10, drop_last=False):
        # words_batch/tags_batch: (batch_size, seq_len)
        # print (f'words_batch: {words_batch}')
        # print (f'tags_batch: {tags_batch}')
        # print (f'words_batch: {words_batch.shape}, tags_batch: {tags_batch.shape}')
        # input ()
        cnt += words_batch.shape[0]
    print (f'trainset: {cnt}')

    cnt = 0
    for words_batch, tags_batch in dataset.testset(batch_size=10, drop_last=False):
        # words_batch/tags_batch: (batch_size, seq_len)
        # print (f'words_batch: {words_batch}')
        # print (f'tags_batch: {tags_batch}')
        # print (f'words_batch: {words_batch.shape}, tags_batch: {tags_batch.shape}')
        # input ()
        cnt += words_batch.shape[0]
    print (f'trainset: {cnt}')
