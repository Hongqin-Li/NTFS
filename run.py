
import torch


# MRC
from data.cmrc.dataset_bert import Dataset as Dataset_cmrc
from data.drcd.dataset_bert import Dataset as Dataset_drcd
# NER
from data.people_daily.dataset_bert import Dataset as Dataset_peopledaily
from data.msra.dataset_bert import Dataset as Dataset_msra
# NLI
from data.xnli.dataset_bert import Dataset as Dataset_xnli
# SC
from data.weibo.dataset_bert import Dataset as Dataset_weibo
from data.chnsenticorp.dataset_bert import Dataset as Dataset_chnsenticorp
# SPM
from data.lcqmc.dataset_bert import Dataset as Dataset_lcqmc
from data.bq_corpus.dataset_bert import Dataset as Dataset_bqcorpus
# DC
from data.thucnews.dataset_bert import Dataset as Dataset_thucnews


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

w2i, i2w = parse_dict('../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt')

def word_to_idx(w):
    return w2i.get(w, w2i['[UNK]'])

dataset_cmrc = Dataset_cmrc(train_file='data/cmrc/cmrc2018/squad-style-data/cmrc2018_train.json', 
                            dev_file='data/cmrc/cmrc2018/squad-style-data/cmrc2018_dev.json', 
                            test_file='data/cmrc/cmrc2018/squad-style-data/cmrc2018_trial.json', 
                            word_to_idx=word_to_idx)

dataset_drcd = Dataset_drcd(train_file='data/drcd/DRCD/DRCD_training.json', 
                            dev_file='data/drcd/DRCD/DRCD_dev.json', 
                            test_file='data/drcd/DRCD/DRCD_test.json', 
                            word_to_idx=word_to_idx)

dataset_peopledaily = Dataset_peopledaily(train_file='data/people_daily/train.txt', 
                                          dev_file='data/people_daily/dev.txt', 
                                          test_file='data/people_daily/test.txt', 
                                          word_to_idx=word_to_idx)

dataset_msra = Dataset_msra(train_file='data/msra/msra_train_bio.txt', 
                            test_file='data/msra/msra_test_bio.txt', 
                            word_to_idx=word_to_idx, 
                            split=0.9)

dataset_xnli = Dataset_xnli(train_file='data/xnli/XNLI-MT-1.0/multinli/multinli.train.zh.tsv', 
                            dev_file='data/xnli/XNLI-1.0/xnli.dev.tsv', 
                            test_file='data/xnli/XNLI-1.0/xnli.test.tsv', 
                            word_to_idx=word_to_idx)


dataset_chnsenticorp = Dataset_chnsenticorp(train_file='data/chnsenticorp/train.tsv', 
                                            dev_file='data/chnsenticorp/dev.tsv', 
                                            test_file='data/chnsenticorp/test.tsv', 
                                            word_to_idx=word_to_idx)

dataset_weibo = Dataset_weibo(raw_file='data/weibo/weibo_senti_100k.csv', 
                              train_file='data/weibo/train.tsv', 
                              dev_file='data/weibo/dev.tsv', 
                              test_file='data/weibo/test.tsv', 
                              word_to_idx=word_to_idx)

dataset_lcqmc = Dataset_lcqmc(train_file='data/lcqmc/train.txt', 
                            dev_file='data/lcqmc/dev.txt', 
                            test_file='data/lcqmc/test.txt', 
                            word_to_idx=word_to_idx)

dataset_bqcorpus = Dataset_bqcorpus(train_file='data/bq_corpus/train.csv', 
                                    dev_file='data/bq_corpus/dev.csv', 
                                    test_file='data/bq_corpus/test.csv', 
                                    word_to_idx=word_to_idx)

dataset_thucnews = Dataset_thucnews(train_file='data/thucnews/cnews/cnews.train.txt',  
                                    dev_file='data/thucnews/cnews/cnews.val.txt', 
                                    test_file='data/thucnews/cnews/cnews.test.txt', 
                                    word_to_idx=word_to_idx)

# config = BertConfig(json_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json')
config = BertConfig(json_path='../bert_checkpoints/bert_toy_config.json')


def run_sequence_classification(dataset, save_path, batch_size=10):

    # model = BertForSequenceClassification(num_classes=dataset.num_classes, config=config, tf_checkpoint_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')

    model = BertForSequenceClassification(num_classes=dataset.num_classes, config=config)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    if torch.cuda.is_available():
        model = model.cuda()
        dataset.use_gpu = True

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      metrics=accuracy_score, 
                      dataset=dataset, 
                      save_path=save_path)

    trainer.train(batch_size=batch_size)

def run_sequence_labeling(dataset, save_path, batch_size=10):

    # model = BertForSequenceClassification(num_classes=dataset.num_classes, config=config, tf_checkpoint_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')

    model = BertForSequenceLabeling(num_classes=dataset.num_classes, config=config)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    if torch.cuda.is_available():
        model = model.cuda()
        dataset.use_gpu = True

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      # NOTE should try other
                      metrics=ner_f1_score, 
                      dataset=dataset, 
                      save_path=save_path)

    trainer.train(batch_size=batch_size)


def run_question_answering(dataset, save_path, batch_size=10):
    
    # model = BertForSequenceClassification(config=config, tf_checkpoint_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')

    model = BertForQuestionAnswering(config=config)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    if torch.cuda.is_available():
        model = model.cuda()
        dataset.use_gpu = True

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      # NOTE try other metrics
                      metrics=qa_f1_score, 
                      dataset=dataset, 
                      save_path=save_path)

    trainer.train(batch_size=batch_size)


if __name__ == '__main__':

    pass
    

