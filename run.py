import argparse
import torch

from data.cmrc.dataset_bert import Dataset as Dataset_cmrc
from data.drcd.dataset_bert import Dataset as Dataset_drcd
from data.people_daily.dataset_bert import Dataset as Dataset_peopledaily
from data.msra.dataset_bert import Dataset as Dataset_msra
from data.xnli.dataset_bert import Dataset as Dataset_xnli
from data.weibo.dataset_bert import Dataset as Dataset_weibo
from data.chnsenticorp.dataset_bert import Dataset as Dataset_chnsenticorp
from data.lcqmc.dataset_bert import Dataset as Dataset_lcqmc
from data.bq_corpus.dataset_bert import Dataset as Dataset_bqcorpus
from data.thucnews.dataset_bert import Dataset as Dataset_thucnews

from models.bert_models import BertForSequenceClassification, BertForSequenceLabeling, BertForQuestionAnswering
from models.bert import BertConfig
from trainer import Trainer
from optim import WarmupOptimizer, AdamW
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


config = BertConfig(json_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json')
# config = BertConfig(json_path='../bert_checkpoints/chinese_L-12_H-768_A-12/bert_config.json')
# config = BertConfig(json_path='../bert_checkpoints/bert_toy_config.json')

def run_sequence_classification(dataset, save_path, batch_size, lr, epochs=3, warmup_portion=0.1, num_save_steps=None, mini_batch_size=8):

    total_train_samples = epochs * dataset.num_train_samples
    num_warmup_steps = int(total_train_samples / batch_size * warmup_portion)
    
    print (f'[Warmup steps] {num_warmup_steps}')
    print (f'[Learing Rate] {lr}')

    model = BertForSequenceClassification(num_classes=dataset.num_classes, config=config, tf_checkpoint_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')
    # model = BertForSequenceClassification(num_classes=dataset.num_classes, config=config)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=lr, 
                      eps=1e-6)

    optimizer = WarmupOptimizer(optimizer=optimizer, num_warmup_steps=num_warmup_steps, lr=lr)

    if torch.cuda.is_available():
        model = model.cuda()
        dataset.use_gpu = True

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      metrics=accuracy_score, 
                      dataset=dataset, 
                      save_path=save_path, 
                      num_save_steps=num_save_steps)

    trainer.train(batch_size=batch_size, mini_batch_size=mini_batch_size)

def run_sequence_labeling(dataset, save_path, batch_size, lr, epochs=3, warmup_portion=0.1, num_save_steps=None, mini_batch_size=8):

    total_train_samples = epochs * dataset.num_train_samples
    num_warmup_steps = int(total_train_samples / batch_size * warmup_portion)
    
    print (f'[warmup steps] {num_warmup_steps}')
    print (f'[learing rate] {lr}')

    model = BertForSequenceLabeling(num_classes=dataset.num_classes, config=config, tf_checkpoint_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')
    # model = bertforsequenceClassification(num_classes=dataset.num_classes, config=config)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=lr, 
                      eps=1e-6)

    optimizer = WarmupOptimizer(optimizer=optimizer, num_warmup_steps=num_warmup_steps, lr=lr)

    if torch.cuda.is_available():
        model = model.cuda()
        dataset.use_gpu = True

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      # FIXME should try other metrics
                      metrics=ner_f1_score, 
                      dataset=dataset, 
                      save_path=save_path, 
                      num_save_steps=num_save_steps)

    trainer.train(batch_size=batch_size, mini_batch_size=mini_batch_size)

def run_question_answering(dataset, save_path, batch_size, lr, epochs=3, warmup_portion=0.1, num_save_steps=None, mini_batch_size=8):

    total_train_samples = epochs * dataset.num_train_samples
    num_warmup_steps = int(total_train_samples / batch_size * warmup_portion)
    
    print (f'[Warmup steps] {num_warmup_steps}')
    print (f'[Learing Rate] {lr}')

    model = BertForQuestionAnswering(config=config, tf_checkpoint_path='../bert_checkpoints/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')
    # model = BertForSequenceClassification(num_classes=dataset.num_classes, config=config)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=lr, 
                      eps=1e-6)

    optimizer = WarmupOptimizer(optimizer=optimizer, num_warmup_steps=num_warmup_steps, lr=lr)

    if torch.cuda.is_available():
        model = model.cuda()
        dataset.use_gpu = True

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      # FIXME should try other metrics
                      metrics=qa_em_score, 
                      dataset=dataset, 
                      save_path=save_path, 
                      num_save_steps=num_save_steps)

    trainer.train(batch_size=batch_size, mini_batch_size=mini_batch_size)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', required=False, default=0)
    parser.add_argument('-t', '--task', required=True)
    parser.add_argument('-b', '--batch_size', required=False, default=64)
    parser.add_argument('-l', '--learning_rate', required=False, default=2e-5)
    # parser.add_argument('-mb', '--mini_batch_size', required=False, default=8)

    args = parser.parse_args()

    device = int(args.device)
    task = args.task
    batch_size = int(args.batch_size)
    # mini_batch_size = int(args.mini_batch_size)
    lr = float(args.learning_rate)

    torch.cuda.set_device(device)

    save_path = f'checkpoints/{task}_lr{lr}_b{batch_size}_{device}.pt'

    if task == 'cmrc':
        dataset = Dataset_cmrc(train_file='data/cmrc/cmrc2018/squad-style-data/cmrc2018_train.json', 
                               dev_file='data/cmrc/cmrc2018/squad-style-data/cmrc2018_dev.json', 
                               test_file='data/cmrc/cmrc2018/squad-style-data/cmrc2018_trial.json', 
                               word_to_idx=word_to_idx, 
                               max_seq_len=512)

        # mini-batch 1 to prevent OOM
        run_question_answering(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path, num_save_steps=50, mini_batch_size=1)

    elif task == 'drcd':
        dataset = Dataset_drcd(train_file='data/drcd/DRCD/DRCD_training.json', 
                              dev_file='data/drcd/DRCD/DRCD_dev.json', 
                              test_file='data/drcd/DRCD/DRCD_test.json', 
                              word_to_idx=word_to_idx, 
                              max_seq_len=512)

        # mini-batch 1 to prevent OOM
        run_question_answering(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path, num_save_steps=100, mini_batch_size=1)

    elif task == 'people_daily' or task == 'peopledaily':
        dataset = Dataset_peopledaily(train_file='data/people_daily/train.txt', 
                                      dev_file='data/people_daily/dev.txt', 
                                      test_file='data/people_daily/test.txt', 
                                      word_to_idx=word_to_idx, 
                                      max_seq_len=256)
        # FIXME num_save_steps
        run_sequence_labeling(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path, num_save_steps=100)

    elif task == 'msra' or task == 'msra_ner':

        dataset = Dataset_msra(train_file='data/msra/msra_train_bio.txt', 
                               test_file='data/msra/msra_test_bio.txt', 
                               word_to_idx=word_to_idx, 
                               split=0.9, 
                               max_seq_len=256)
        run_sequence_labeling(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path, num_save_steps=100)

    # OK
    elif task == 'xnli':
        dataset = Dataset_xnli(train_file='data/xnli/XNLI-MT-1.0/multinli/multinli.train.zh.tsv', 
                               dev_file='data/xnli/XNLI-1.0/xnli.dev.tsv', 
                               test_file='data/xnli/XNLI-1.0/xnli.test.tsv', 
                               word_to_idx=word_to_idx, 
                               max_seq_len=128)

        run_sequence_classification(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path, num_save_steps=100)

    # OK
    elif task == 'chnsenticorp':

        dataset = Dataset_chnsenticorp(train_file='data/chnsenticorp/train.tsv', 
                                       dev_file='data/chnsenticorp/dev.tsv', 
                                       test_file='data/chnsenticorp/test.tsv', 
                                       word_to_idx=word_to_idx, 
                                       max_seq_len=256)

        run_sequence_classification(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path)

    # OK
    elif task == 'weibo' or task == 'sina_weibo':

        dataset = Dataset_weibo(raw_file='data/weibo/weibo_senti_100k.csv', 
                                train_file='data/weibo/train.tsv', 
                                dev_file='data/weibo/dev.tsv', 
                                test_file='data/weibo/test.tsv', 
                                word_to_idx=word_to_idx, 
                                max_seq_len=128)

        run_sequence_classification(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path, num_save_steps=100)

    # OK
    elif task == 'lcqmc':

        dataset = Dataset_lcqmc(train_file='data/lcqmc/train.txt', 
                                dev_file='data/lcqmc/dev.txt', 
                                test_file='data/lcqmc/test.txt', 
                                word_to_idx=word_to_idx, 
                                max_seq_len=128)

        run_sequence_classification(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path, num_save_steps=100)

    # OK
    elif task == 'bq_corpus' or task == 'bqcorpus':

        dataset = Dataset_bqcorpus(train_file='data/bq_corpus/train.csv', 
                                   dev_file='data/bq_corpus/dev.csv', 
                                   test_file='data/bq_corpus/test.csv', 
                                   word_to_idx=word_to_idx, 
                                   max_seq_len=128)

        run_sequence_classification(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path)
    
    # OK
    elif task == 'thucnews':

        dataset = Dataset_thucnews(raw_train_file='data/thucnews/cnews/cnews.train.txt',  
                                   train_file='data/thucnews/cnews/cnews.train.shuffled.txt',  
                                   dev_file='data/thucnews/cnews/cnews.val.txt', 
                                   test_file='data/thucnews/cnews/cnews.test.txt', 
                                   word_to_idx=word_to_idx, 
                                   max_seq_len=512)

        # mini-batch 1 to prevent OOM
        run_sequence_classification(dataset, batch_size=batch_size, lr=lr, epochs=2, save_path=save_path, num_save_steps=100, mini_batch_size=1)
        
    

