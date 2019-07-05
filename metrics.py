import torch

from collections import Counter

# TODO test all functions below!

def accuracy_score(model, dataset):

    score, total = 0, 0
    for batch in dataset(batch_size=100):

        # NOTE model should support "predict" method
        pred = model.predict(batch.input) # (batch_size)
        target = batch.target # (batch_size)

        score += torch.sum(pred == target).item()
        total += target.shape[0]
     
    return score / total

def qa_em_score(model, dataset):
    # EM(Exact Match) Metrics for Question Answering tasks
    # EM: both start index and end index are identical to those of target

    score, total = 0, 0
    for batch in dataset:

        pred_start_idxs, pred_end_idxs = model.predict(batch.input)
        start_idxs, end_idxs = batch.target
        # all of shape (batch_size)

        score = torch.sum((pred_start_idxs == start_idxs) * (pred_end_idxs == end_idxs)).item()
        total += start_idxs.shape[0]

    return score / total
    

def qa_f1_score(model, dataset):
    # Details can be found in SQUAD paper
    score, total = 0, 0

    for batch in dataset:

        pred_start_idxs, pred_end_idxs = model.predict(batch.input)
        start_idxs, end_idxs = batch.target
        # all of shape (batch_size)
        
        # NOTE dataset should support this
        raw_docs = batch.raw_documents
        # [doc1, doc2, ...]: list of raw document strings, used to get the original answer string by predicted answer span 

        for si, ei, psi, pei, doc in zip(start_idxs.tolist(), end_idxs.tolist(), pred_start_idxs.tolist(), pred_end_idxs.tolist(), raw_docs):
            ans = [word for word in doc[si: ei + 1]]
            pred_ans = [word for word in doc[psi: pei + 1]]

            num_same = sum((Counter(ans) & Counter(pred_ans)).values())
            precision = num_same / len(pred_ans) if len(pred_ans) > 0 else 0
            recall = num_same / len(ans) if len(ans) > 0 else 0
            
            score += 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        total += start_idxs.shape[0]

    return score / total

def get_entries(tags):
    # tags: list of tags, e.g. ['O', 'B-PER', 'I-PER', 'B-ORG']
    # return: list of entries, each of which is a tuple of (catagory, start index, end index), e.g. [('PER', 1, 2), ('ORG', 3, 3)]

    entries = []
    current_catagory = ''
    start_idx = 0

    for i, tag in enumerate(tags + ['O']):

        tag = tag.split('-')
        catagory = tag[-1] # PER, ORG, LOC, O
        prefix = tag[0] # B, I, O
        
        # End of span
        if prefix != 'I' and current_catagory != '':
            entries.append((curren_catagory, start_idx, i - 1))
            current_catagory = ''

        # A tag span can only start with B-xxx
        if prefix == 'B':
            current_catagory, start_idx = catagory, i
            
    return entries

def precision_score(pred_tags, true_tags):

    pred_entries = set(get_entries(pred_tags))
    true_entries = set(get_entries(true_tags))

    num_correct = len(pred_entries & true_entries)
    num_pred = len(pred_entries)

    return num_correct / num_pred if num_pred > 0 else 0
    
def recall_score(pred_tags, true_tags):
    
    pred_entries = set(get_entries(pred_tags))
    true_entries = set(get_entries(true_tags))

    num_correct = len(pred_entries & true_entries)
    num_true = len(true_entries)

    return num_correct / num_true if num_true > 0 else 0

def f1_score(pred_tags, true_tags):

    pred_entries = set(get_entries(pred_tags))
    true_entries = set(get_entries(true_tags))

    num_correct = len(pred_entries & true_entries)
    num_true = len(true_entries)
    num_pred = len(pred_entries)

    precision = num_correct / num_pred if num_pred > 0 else 0
    recall = num_correct / num_true if num_true > 0 else 0

    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


def ner_precision_score(model, dataset):

    score, total = 0, 0
    for batch in dataset:

        pred = model(batch.input)
        # (batch_size, seq_len)

        for inp, target in zip(pred.tolist(), batch.target.tolist()):
            
            # NOTE dataset should support this, including padding index
            pred_tags = [batch.idx_to_tag(i) for i in inp]
            true_tags = [batch.idx_to_tag(i) for i in target]
            score += precision_score(pred_tags, true_tags)

        total += pred.shape[0]

    return score / total

# Similar to above
def ner_recall_score(model, dataset):
    score, total = 0, 0
    for batch in dataset:
        pred = model(batch.input)
        for inp, target in zip(pred.tolist(), batch.target.tolist()):
            pred_tags = [batch.idx_to_tag(i) for i in inp]
            true_tags = [batch.idx_to_tag(i) for i in target]
            score += recall_score(pred_tags, true_tags)
        total += pred.shape[0]
    return score / total

# Similar to above
def ner_f1_score(model, dataset):
    score, total = 0, 0
    for batch in dataset:
        pred = model(batch.input)
        for inp, target in zip(pred.tolist(), batch.target.tolist()):
            pred_tags = [batch.idx_to_tag(i) for i in inp]
            true_tags = [batch.idx_to_tag(i) for i in target]
            score += f1_score(pred_tags, true_tags)
        total += pred.shape[0]
    return score / total


if __name__ == '__main__':  
    pass



