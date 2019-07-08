import json
from copy import deepcopy
import torch
import torch.nn as nn

from collections import namedtuple

# Can be simply regarded as a object, called "Batch", having attributes "input" and "target"
Batch = namedtuple('Batch', 'input target raw_documents')

punctuations = {'，', '。', '？', '！', '；', '、', ',', '.', ';'}

def is_punctuation(w):
    if w in punctuations: return True
    else: return False
    
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

    # FIXME answer len no use
    def __init__(self, train_file, dev_file, test_file, word_to_idx, use_gpu=False, max_seq_len=512, max_answer_len=30, max_query_len=64):
        self.use_gpu = use_gpu

        self.max_seq_len = max_seq_len
        self.max_query_len = max_query_len
        self.max_answer_len = max_answer_len

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.word_to_idx = word_to_idx

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
   
    def str_to_tensor(self, s):
        # s: string
        # return: 1-d long tensor of shape (seq_len)

        # TODO prune long str
        return torch.LongTensor([self.word_to_idx(c) for c in s])
    
    def pad_sequence(self, s):
        # TODO pad to 512?
        return nn.utils.rnn.pad_sequence(s, batch_first=True)

    def samples(self, file_path):

        with open(file_path, 'r') as f:
            data = json.load(f)
            data = data['data']

            for d in data:

                for p in d['paragraphs']:
                    raw_doc = p['context']

                    for qa in p['qas']:
                        quest = qa['question']

                        a_set = set()
                    
                        for ans in qa['answers']:
                            # devset with duplicate answers provided...
                            # only choose the first answer
                       
                            if len(a_set) != 0: break
                            a_set.add(ans['text'])

                            ans_start_idx = raw_doc.find(ans['text'])
                            ans_end_idx = ans_start_idx + len(ans['text']) - 1

                            try:
                                assert ans_start_idx >= 0 
                            except:
                                print ('Warning: cannot find answer in the document')
                                print (f'doc: {doc}')
                                print (f'query: {query}')
                                print (f'ans: {ans["text"]}')
                                raise


                            # bert-like preprocess
                            # Truncate query
                            query_len = min(self.max_query_len, len(quest))
                            query = quest[: query_len]

                            # print (f'raw_doc: {doc}')

                            doc = raw_doc
                            # Truncate doc
                            max_doc_len = self.max_seq_len - 3 - query_len # 3 for [CLS] query [SEP] doc [SEP]
                            raw_doc_len = len(raw_doc)

                            if raw_doc_len > max_doc_len:
                                # print ('Warning: exceed max length!')

                                middle_idx = (ans_start_idx + ans_end_idx) // 2
                                truncated_doc_start_idx = max(0, middle_idx - max_doc_len // 2)
                                truncated_doc_end_idx = min(raw_doc_len - 1, truncated_doc_start_idx + max_doc_len - 1)

                                # Expect to find the "perfect" doc span that has whole sentences
                                if truncated_doc_start_idx != 0:
                                    for i in range(truncated_doc_start_idx, ans_start_idx + 1):
                                        if is_punctuation(doc[i]) and i + 1 <= ans_start_idx: 
                                            truncated_doc_start_idx = i + 1
                                            break

                                if truncated_doc_end_idx != raw_doc_len - 1:
                                    for i in range(truncated_doc_end_idx, ans_end_idx - 1, -1):
                                        if is_punctuation(doc[i]) and i - 1 >= ans_end_idx:
                                            truncated_doc_end_idx = i - 1
                                            break

                                doc = raw_doc[truncated_doc_start_idx: truncated_doc_end_idx + 1]
                                ans_start_idx -= truncated_doc_start_idx 
                                ans_end_idx -= truncated_doc_start_idx

                            try:
                                assert 0 <= ans_start_idx <= ans_end_idx < len(doc)
                                assert len(doc) + len(query) + 3 <= self.max_seq_len
                            except:
                                print (f'si[{ans_start_idx}]: {doc[ans_start_idx]}, ei[{ans_end_idx}]: {doc[ans_end_idx - 1]}')
                                print (f'doc[{len(doc)}]: {doc}\nquery[{len(quest)}]: {quest}')
                                raise

                            # print (f'raw_doc: {raw_doc}\ndoc: {doc}\nquery: {query}\nanswer: {doc[ans_start_idx: ans_end_idx + 1]}')
                            # input ()
                            
                            yield doc, query, ans_start_idx, ans_end_idx

   
    def sample_batches(self, file_path, batch_size=1, drop_last=False):
        # drop_last: drop the last incomplete batch if True
        cnt = 0

        # Input
        token_idxs_batch, token_type_idxs_batch, mask_batch = [], [], [] 

        # Target
        start_idx_batch = [] # (batch_size)
        end_idx_batch = [] # (batch_size)

        # Raw documents
        raw_doc_batch = [] # List of string: [doc_{1}, ..., doc_{i}, ..., doc_{batch_size}]

        for doc, query, si, ei in self.samples(file_path):

            token_idxs, token_type_idxs, mask = parse_sentence_pair(query, doc, self.word_to_idx, self.max_seq_len)        
            # all (batch_size, seq_len)

            token_idxs = torch.LongTensor(token_idxs)
            token_type_idxs = torch.LongTensor(token_type_idxs)
            mask = torch.LongTensor(mask)
            si = torch.LongTensor([si])
            ei = torch.LongTensor([ei])

            if self.use_gpu:
                token_idxs, token_type_idxs, mask, si, ei = token_idxs.cuda(), token_type_idxs.cuda(), mask.cuda(), si.cuda(), ei.cuda()

            token_idxs_batch.append(token_idxs)
            token_type_idxs_batch.append(token_type_idxs)
            mask_batch.append(mask)
            start_idx_batch.append(si)
            end_idx_batch.append(ei)

            raw_doc_batch.append(doc)

            cnt += 1

            if cnt >= batch_size:

                yield Batch(input=(self.pad_sequence(token_idxs_batch), 
                                   self.pad_sequence(token_type_idxs_batch), 
                                   self.pad_sequence(mask_batch)), 
                            target=(torch.cat(start_idx_batch), torch.cat(end_idx_batch)), 
                            raw_documents=raw_doc_batch)

                token_idxs_batch, token_type_idxs_batch, mask_batch, start_idx_batch, end_idx_batch, raw_doc_batch = [], [], [], [], [], []
                cnt = 0

        if cnt > 0 and not drop_last:
            yield Batch(input=(self.pad_sequence(token_idxs_batch), 
                               self.pad_sequence(token_type_idxs_batch), 
                               self.pad_sequence(mask_batch)), 
                        target=(torch.cat(start_idx_batch), torch.cat(end_idx_batch)), 
                        raw_documents=raw_doc_batch)


if __name__ == '__main__': 
    # Usage
    train_file = 'cmrc2018/squad-style-data/cmrc2018_train.json'
    dev_file = 'cmrc2018/squad-style-data/cmrc2018_dev.json'
    test_file = 'cmrc2018/squad-style-data/cmrc2018_trial.json'

    # maps word to index, 0 for OOV, e.g. word_to_idx('a') = 1
    def word_to_idx(w):
        w2i = {'a': 1}
        return w2i.get(w, 0)

    dataset = Dataset(train_file=train_file, dev_file=dev_file, test_file=test_file, word_to_idx=word_to_idx)

    print (f'trainset: {dataset.num_train_samples}')
    print (f'devset: {dataset.num_dev_samples}')
    print (f'testset: {dataset.num_test_samples}')

    cnt = 0
    for (token_idxs, token_type_idxs, masks), (start_idxs, end_idxs), raw_docs in dataset.trainset(batch_size=100, drop_last=False):
        # print (f'doc: {doc.shape}, quest: {quest.shape}, start_idx: {start_idx.shape}, end_idx: {end_idx.shape}')
        # print (quest)
        # print (len(raw_docs))
        # input ()
        cnt += token_idxs.shape[0]

    print (f'trainset: {cnt}')
    

    cnt = 0
    for (token_idxs, token_type_idxs, masks), (start_idxs, end_idxs), raw_docs in dataset.devset(batch_size=100, drop_last=False):
        # print (f'doc: {doc.shape}, quest: {quest.shape}, start_idx: {start_idx.shape}, end_idx: {end_idx.shape}')
        # print (quest)
        # input ()
        cnt += token_idxs.shape[0]

    print (f'devset: {cnt}')

    cnt = 0
    for (token_idxs, token_type_idxs, masks), (start_idxs, end_idxs), raw_docs in dataset.testset(batch_size=100, drop_last=False):
        # print (f'doc: {doc.shape}, quest: {quest.shape}, start_idx: {start_idx.shape}, end_idx: {end_idx.shape}')
        # print (quest)
        # input ()
        cnt += token_idxs.shape[0]

    print (f'testset: {cnt}')

