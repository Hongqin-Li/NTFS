import torch

from collections import Counter

# TODO test all functions below!

def accuracy_score(model, dataset, batch_size=8):

    score, total = 0, 0
    for batch in dataset(batch_size=batch_size):

        # NOTE model should support "predict" method
        pred = model.predict(batch.input) # (batch_size)
        target = batch.target # (batch_size)

        score += torch.sum(pred == target).item()
        total += target.shape[0]
     
    return score / total

def qa_em_score(model, dataset, batch_size=8):
    # EM(Exact Match) Metrics for Question Answering tasks
    # EM: both start index and end index are identical to those of target

    score, total = 0, 0
    for batch in dataset(batch_size=batch_size):

        pred_start_idxs, pred_end_idxs = model.predict(batch.input)
        start_idxs, end_idxs = batch.target
        # all of shape (batch_size)

        score += torch.sum((pred_start_idxs == start_idxs) * (pred_end_idxs == end_idxs)).item()
        total += start_idxs.shape[0]

    return score / total
    

def qa_f1_score(model, dataset, batch_size=8):
    # Details can be found in SQUAD paper
    score, total = 0, 0

    for batch in dataset(batch_size=batch_size):

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


def parse_padded_batch(padded_pred, padded_target, idx_to_tag, padding_idx=-1):
    # padded_pred:   (batch_size, seq_len)
    # padded_target: (batch_size, seq_len)
    
    pred = []
    target = []

    for padded_p, padded_t in zip(padded_pred, padded_target):
        # both of shape (seq_len)

        p, t = [], []

        for pi, ti in zip(padded_p, padded_t):
            if ti == padding_idx: continue
            p.append(idx_to_tag(pi))
            t.append(idx_to_tag(ti))

        pred.append(p)
        target.append(t)
                
    return pred, target

def get_entries(tags):
    # tags: list of tags, e.g. ['O', 'B-PER', 'I-PER', 'B-ORG']
    # return: list of entries, each of which is a tuple of (catagory, start index, end index), e.g. [('PER', 1, 2), ('ORG', 3, 3)]

    entries = []
    current_catagory = ''
    start_idx = 0

    # for nested list
    if any(isinstance(s, list) for s in tags):
        tags = [item for sublist in tags for item in sublist + ['O']]

    for i, tag in enumerate(tags + ['O']):

        tag = tag.split('-')
        catagory = tag[-1] # PER, ORG, LOC, O
        prefix = tag[0] # B, I, O
        
        # End of span
        if prefix != 'I' and current_catagory != '':
            entries.append((current_catagory, start_idx, i - 1))
            current_catagory = ''

        # A tag span can only start with B-xxx
        if prefix == 'B':
            current_catagory, start_idx = catagory, i
            
    return entries

def ner_score(pred_tags, target_tags, score_type):

    pred_entries = set(get_entries(pred_tags))
    target_entries = set(get_entries(target_tags))

    num_correct = len(pred_entries & target_entries)
    num_target = len(target_entries)
    num_pred = len(pred_entries)

    precision = num_correct / num_pred if num_pred > 0 else 0
    recall = num_correct / num_target if num_target > 0 else 0

    print (f'precision: {precision}, recall: {recall}')

    if score_type == 'precision':
        return precision

    elif score_type == 'recall':
        return recall

    elif score_type == 'f1':
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    else:
        print (f'Warning: {score_type} not supported!')


def ner_precision_score(model, dataset, batch_size=8):

    preds = []
    targets = []

    for batch in dataset(batch_size=batch_size):

        pred = model.predict(batch.input)
        # (batch_size, seq_len)

        pred, target = parse_padded_batch(pred, batch.target, batch.idx_to_tag)

        preds += pred
        targets += target


    return ner_score(preds, targets, score_type='precision')


def ner_recall_score(model, dataset, batch_size=8):

    preds = []
    targets = []

    for batch in dataset(batch_size=batch_size):

        pred = model.predict(batch.input)
        # (batch_size, seq_len)

        pred, target = parse_padded_batch(pred, batch.target, batch.idx_to_tag)

        preds += pred
        targets += target

    return ner_score(preds, targets, score_type='recall')


def ner_f1_score(model, dataset, batch_size=8):

    preds = []
    targets = []

    for batch in dataset(batch_size=batch_size):

        pred = model.predict(batch.input)
        # (batch_size, seq_len)

        pred, target = parse_padded_batch(pred, batch.target, batch.idx_to_tag)

        preds += pred
        targets += target

    return ner_score(preds, targets, score_type='f1')


if __name__ == '__main__':  
    # Test
    y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    y_target = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]

    print (ner_score(y_pred, y_target, score_type='precision'))
    print (ner_score(y_pred, y_target, score_type='recall'))
    print (ner_score(y_pred, y_target, score_type='f1'))




