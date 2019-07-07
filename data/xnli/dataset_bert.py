import json
import torch
import torch.nn as nn

from collections import namedtuple

# Can be simply regarded as a object, called "Batch", having attributes "input" and "target"
Batch = namedtuple('Batch', 'input target')

# Bert-like preprocessing
def parse_sentence_pair(sent1, sent2, word_to_idx, max_seq_len):

    token_idxs, token_type_idxs, mask = [word_to_idx('[CLS]')], [0], [1] # (seq_len)

    len1, len2 = len(sent1), len(sent2)

    if len1 + len2 > max_seq_len - 3:
        print ('Warning: exceed max_seq_len')
        # Weighted truncate
        len1 = int( len1 / (len1 + len2) * (max_seq_len - 3) )
        len2 = max_seq_len - 3 - len1
        sent1 = sent1[: len1]
        sent2 = sent2[: len2]

    assert len(sent1) + len(sent2) <= max_seq_len - 3

    token_idxs += [word_to_idx(w) for w in sent1] + [word_to_idx('[SEP]')] + [word_to_idx(w) for w in sent2] + [word_to_idx('[SEP]')]
    token_type_idxs += [0] * (len1 + 1) + [1] * (len2 + 1) # +1 for [SEP] following each sentence
    mask += [1] * (len1 + len2 + 2) # two [SEP] for each sentence

    assert len(token_idxs) == len(token_type_idxs) and len(token_idxs) == len(mask) and len(mask) <= max_seq_len

    return token_idxs, token_type_idxs, mask


class Dataset(): 

    def __init__(self, train_file, dev_file, test_file, word_to_idx, tag_to_idx, max_seq_len=512):
        # word_to_idx/tag_to_idx: both are functions, whose input is a string and output an int
        self.max_seq_len = 512

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx

        self.num_train_samples = 0
        for batch in self.trainset(batch_size=1000):
            self.num_train_samples += batch[-1].shape[0]

        self.num_dev_samples = 0
        for batch in self.devset(batch_size=1000):
            self.num_dev_samples += batch[-1].shape[0]

        self.num_test_samples = 0
        for batch in self.testset(batch_size=1000):
            self.num_test_samples += batch[-1].shape[0]

    def trainset(self, batch_size=1, drop_last=False):
        for batch in self.sample_batches(self.train_file, batch_size=batch_size, drop_last=drop_last):
            yield batch
            
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
        # s: string or list
        # return: 1-d long tensor of shape (seq_len)
        return torch.LongTensor([self.word_to_idx(w) for w in s])

    def pad_sequence(self, s):
        # TODO pad to 512?
        return nn.utils.rnn.pad_sequence(s, batch_first=True)

    def samples(self, file_path):

        with open(file_path, 'r') as f:

            labels = f.readline().strip().split('\t') # Omit tsv header

            is_train = file_path == self.train_file

            for line in f:
                line = line.strip().split('\t')

                # Since the trainset is different from devset and testset
                if is_train:
                    sent1 = line[labels.index('premise')]
                    sent2 = line[labels.index('hypo')]
                    tag = line[labels.index('label')]

                else: 
                    language = line[labels.index('language')]
                    if language != 'zh': continue
                    sent1 = line[labels.index('sentence1')]
                    sent2 = line[labels.index('sentence2')]
                    tag = line[labels.index('gold_label')]

                # sent1, sent2, tag = ''.join(sent1.split()), ''.join(sent2.split()), tag
                # print (sent1, sent2, tag)
                # input ()
                yield ''.join(sent1.split()), ''.join(sent2.split()), tag
                


   
    def sample_batches(self, file_path, batch_size=1, drop_last=False):
        # drop_last: drop the last incomplete batch if True
        cnt = 0

        # Input
        sent1_batch, sent2_batch = [], [] # (batch_size, seq_len)
        token_idxs_batch, token_type_idxs_batch, mask_batch = [], [], []
        # Target
        tag_batch = [] # (batch_size)

        for sent1, sent2, tag in self.samples(file_path):
            # all string-like

            token_idxs, token_type_idxs, mask = parse_sentence_pair(sent1, sent2, self.word_to_idx, self.max_seq_len)        

            token_idxs_batch.append(torch.LongTensor(token_idxs))
            token_type_idxs_batch.append(torch.LongTensor(token_type_idxs))
            mask_batch.append(torch.LongTensor(mask))
            tag_batch.append(self.tag_to_idx(tag))

            cnt += 1

            if cnt >= batch_size:
                yield Batch(input=(self.pad_sequence(token_idxs_batch), self.pad_sequence(token_type_idxs_batch), self.pad_sequence(mask_batch)), target=torch.LongTensor(tag_batch))
                token_idxs_batch, token_type_idxs_batch, mask_batch, tag_batch = [], [], [], []
                cnt = 0

        if cnt > 0 and not drop_last:
            yield Batch(input=(self.pad_sequence(token_idxs_batch), self.pad_sequence(token_type_idxs_batch), self.pad_sequence(mask_batch)), target=torch.LongTensor(tag_batch))


if __name__ == '__main__': 
    # Usage
    train_file = 'XNLI-MT-1.0/multinli/multinli.train.zh.tsv'
    dev_file = 'XNLI-1.0/xnli.dev.tsv'
    test_file = 'XNLI-1.0/xnli.test.tsv'

    def word_to_idx(w):
        w2i = {'当': 1, '希': 2}
        return w2i.get(w, 0)

    def tag_to_idx(tag):
        t2i = {'neutral':0, 'entailment': 1, 'contradictory': 2, 'contradiction': 2}
        return t2i[tag]

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
 
    
