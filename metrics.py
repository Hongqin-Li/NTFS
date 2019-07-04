import torch

from collections import Counter

def classification_metrics(model, dataset):

    score = 0
    cnt = 0

    for batch in dataset(batch_size=100):

        # NOTE model should support "predict" method
        pred = model.predict(batch.input) # (batch_size)
        target = batch.target # (batch_size)

        score += torch.sum(pred == target).item()
        cnt += target.shape[0]
     
    return score / cnt

def qa_em_metrics(model, dataset):
    # EM(Exact Match) Metrics for Question Answering tasks
    # EM: both start index and end index are identical to those of target
    
    score = 0
    total = 0
    for batch in dataset:

        pred_start_idxs, pred_end_idxs = model.predict(batch.input)
        start_idxs, end_idxs = batch.target
        # all of shape (batch_size)

        score = torch.sum((pred_start_idxs == start_idxs) * (pred_end_idxs == end_idxs)).item()
        total += start_idxs.shape[0]

    return score / total
    

def qa_f1_metrics(model, dataset):
    # Details can be found in SQUAD paper
    score = 0
    total = 0

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
            precision = num_same / len(pred_ans)
            recall = num_same / len(ans)
            
            score += 2 * precision * recall / (precision + recall)

        total += start_idxs.shape[0]

    return score / total



if __name__ == '__main__':  
    pass



