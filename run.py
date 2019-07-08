
import torch
from data.sina_weibo.dataset_bert import Dataset

from models.bert_models import BertForSequenceClassification, BertForSequenceLabeling, BertForQuestionAnswering
from models.bert import BertConfig
from trainer import Trainer
from optim import AdamW
from metrics import accuracy_score, qa_em_score, qa_f1_score, ner_precision_score, ner_recall_score, ner_f1_score

def parse_dict(dict_path):
    w2i, i2w = {}, {}
    with open(dict_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            w = line.strip()
            w2i[w] = i 
            i2w[i] = w
    return w2i, i2w


def run_sina_weibo():
    
    raw_file = 'data/sina_weibo/weibo_senti_100k.csv'
    train_file = 'data/sina_weibo/train.tsv'
    dev_file = 'data/sina_weibo/dev.tsv'
    test_file = 'data/sina_weibo/test.tsv'

    w2i, i2w = parse_dict('../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt')

    def word_to_idx(w):
        return w2i.get(w, w2i['[UNK]'])

    dataset = Dataset(raw_file=raw_file, 
                      train_file=train_file, 
                      dev_file=dev_file, 
                      test_file=test_file, 
                      word_to_idx=word_to_idx)

    config = BertConfig(json_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json')
    model = BertForSequenceClassification(num_classes=2, config=config, tf_checkpoint_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')


    # config = BertConfig(json_path='../bert_checkpoints/bert_toy_config.json')
    # model = BertForSequenceClassification(num_classes=2, config=config)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    if torch.cuda.is_available:
        model = model.cuda()
        dataset.use_gpu = True

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      metrics=accuracy_score, 
                      dataset=dataset, 
                      save_path='./checkpoints/sina_weibo.pt')
    trainer.train(batch_size=1)


if __name__ == '__main__':
    run_sina_weibo()
    

