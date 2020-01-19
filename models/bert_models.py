import torch
import torch.nn as nn
import torch.nn.functional as F


def get_position_idxs(token_idxs):

    batch_size, seq_len = token_idxs.shape

    position_idxs = [[i for i in range(seq_len)]] * batch_size

    if token_idxs.is_cuda:
        position_idxs = torch.LongTensor(position_idxs).cuda()
    else:
        position_idxs = torch.LongTensor(position_idxs)

    assert position_idxs.shape[0] == batch_size and position_idxs.shape[1] == seq_len

    return position_idxs

    

class BertForSequenceClassification(nn.Module):

    def __init__(self, num_classes, bert):
        super(BertForSequenceClassification, self).__init__()

        self.num_classes = num_classes
        self.bert_model = bert

        self.linear = nn.Linear(self.bert_model.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        

        '''
        for name, param in self.bert_model.named_parameters():
            print (name, param)
            input ()
        '''

    def forward(self, inp):

        if len(inp) == 3:
            token_idxs, token_type_idxs, masks = inp
            # all of shape (batch_size, seq_len)
        else:
            token_idxs, token_type_idxs = inp
            masks = None
        position_idxs = get_position_idxs(token_idxs)

        _, pooled_first_token_output = self.bert_model(token_idxs, position_idxs, token_type_idxs, masks)
        # (batch_size, hidden_size)

        x = self.linear(self.dropout(pooled_first_token_output))
        # (batch_size, num_classes)
        return x

    def compute_loss(self, pred, target):
        # pred: (batch_size, num_classes)
        # target: (batch_size)
        criterion = nn.CrossEntropyLoss()
        return criterion(pred, target)

    def predict(self, inp):
        x = self.forward(inp)
        # (batch_size, num_classes)

        pred_classes = torch.argmax(F.softmax(x, dim=-1), dim=-1)
        # (batch_size)
        
        return pred_classes

class BertForSequenceLabeling(nn.Module):

    def __init__(self, num_classes, bert):
        super(BertForSequenceLabeling, self).__init__()

        self.num_classes = num_classes
        self.bert_model = bert
        self.linear = nn.Linear(self.bert_model.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inp):

        if len(inp) == 3:
            token_idxs, token_type_idxs, masks = inp
            # all of shape (batch_size, seq_len)
        else:
            token_idxs, token_type_idxs = inp
            masks = None
        position_idxs = get_position_idxs(token_idxs)

        sequence_output, _ = self.bert_model(token_idxs, position_idxs, token_type_idxs, masks)
        # (batch_size, seq_len, hidden_size)

        x = self.linear(self.dropout(sequence_output))
        # (batch_size, seq_len, num_classes)
        return x

    def compute_loss(self, pred, target):
        # pred: (batch_size, seq_len, num_classes)
        # target: (batch_size, seq_len)

        pred = pred[:, 1:-1, :] # ignore [CLS] and [SEP]

        # NOTE Dataset should have a target padding value of -1
        criterion = nn.CrossEntropyLoss(ignore_index=-1) # ignore padding

        return criterion(pred.contiguous().view(-1, self.num_classes), target.contiguous().view(-1))

    def predict(self, inp):
        x = self.forward(inp)
        # (batch_size, seq_len, num_classes)

        x = x[:, 1:-1, :] # ignore [CLS] and [SEP]

        pred_labels = torch.argmax(F.softmax(x, dim=-1), dim=-1)
        # (batch_size, seq_len)
        
        return pred_labels

class BertForQuestionAnswering(nn.Module):

    def __init__(self, bert):
        super(BertForQuestionAnswering, self).__init__()

        self.bert_model = bert
        self.linear = nn.Linear(self.bert_model.hidden_size, 2)

        # NOTE QA-bert has no dropout in official implementation?
        # self.dropout = nn.Dropout(0.1)

    def forward(self, inp):

        if len(inp) == 3:
            token_idxs, token_type_idxs, masks = inp
            # all of shape (batch_size, seq_len)
        else:
            token_idxs, token_type_idxs = inp
            masks = None

        position_idxs = get_position_idxs(token_idxs)

        sequence_output, _ = self.bert_model(token_idxs, position_idxs, token_type_idxs, masks)
        # (batch_size, seq_len, hidden_size)

        # x = self.linear(self.dropout(sequence_output))
        x = self.linear(sequence_output)
        # (batch_size, seq_len, 2)

        return x[:, :, 0], x[:, :, 1]

    def compute_loss(self, pred, target):
        # pred: (batch_size, seq_len), (batch_size, seq_len)
        # target: (batch_size), (batch_size)
        criterion = nn.CrossEntropyLoss()
        return (criterion(pred[0], target[0]) + criterion(pred[1], target[1])) / 2

    def predict(self, inp):
        prob_start, prob_end = self.forward(inp)
        # (batch_size, seq_len)

        pred_start_idxs = torch.argmax(F.softmax(prob_start, dim=-1), dim=-1)
        pred_end_idxs = torch.argmax(F.softmax(prob_end, dim=-1), dim=-1)
        # (batch_size)
        
        return pred_start_idxs, pred_end_idxs


if __name__ == '__main__':

    from bert import BertModel, BertConfig
    from albert import AlbertModel
    # Usage
    # config = BertConfig(json_path='../../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json')
    # bert_model = BertModel(config)
    config = BertConfig(json_path='../checkpoints/albert_tiny_zh/albert_config_tiny.json')
    bert_model = AlbertModel(config, tf_checkpoint_path='../checkpoints/albert_tiny_zh/albert_model.ckpt')

    model_sc = BertForSequenceClassification(num_classes=5, bert=bert_model)
    model_sl = BertForSequenceLabeling(num_classes=5, bert=bert_model)
    model_qa = BertForQuestionAnswering(bert=bert_model)

    token_idxs = torch.LongTensor([[100, 1, 2, 101, 3, 4, 101]])
    token_type_idxs = torch.LongTensor([[ 0 ,  0 ,  0 ,  0 ,  1 ,  1 ,  1 ]])

    inp = token_idxs, token_type_idxs

    out = model_sc(inp)
    pred = model_sc.predict(inp)
    print (f'[Sequence Classification]\nout: {out.shape}, pred: {pred.shape}')

    out = model_sl(inp)
    pred = model_sl.predict(inp)
    print (f'[Sequence Labeling]\nout: {out.shape}, pred: {pred.shape}')

    out_s, out_e = model_qa(inp)
    pred_s, pred_e = model_qa.predict(inp)
    print (f'[Question Answering]\nout_s: {out_s.shape}, out_e: {out_e.shape}, pred_s: {pred_s.shape}, pred_e: {pred_e.shape}')
 





