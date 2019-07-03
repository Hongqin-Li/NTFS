# NTFS: NLP Tester For Slobs





## Downstream Tasks



### Data

- MRC: Evaluation metrics are EM, F1

  - [x] [CMRC2018](https://github.com/ymcui/cmrc2018): only use the first answer of each question; **number of test samples may be wrong?**
  - [x] [DRCD](https://github.com/DRCKnowledgeTeam/DRCD): Traditional Chinese only; only use the first answer of each question.
  - [ ] [CJRC](http://cail.cipsc.org.cn/): only find trainset `small-train-data.json`, no devset and testset, thus omitted.
- NER: Evaluation metrics include *precision*, *Recall* and *F-score*.
  - [x] [People Daily](https://github.com/ProHiryu/bert-chinese-ner/tree/master/data): trainset and devset
  - [x] [MSRA-NER](<https://github.com/OYE93/Chinese-NLP-Corpus>): trainset and testset
- NLI: 
  - [x] [XNLI](https://github.com/facebookresearch/XNLI): testset size not matched, mine is 5k, but the paper only get 2.5k
- SC: , two-class sentence classification, remove stopwords?
  - [x] [ChnSentiCorp](https://github.com/pengming617/bert_classification/)
  - [x] [Sina Weibo](https://github.com/SophonPlus/ChineseNlpCorpus/)
- SPM: 

  - [ ] [LCQMC](http://icrc.hitsz.edu.cn/info/1037/1146.htm), [BQ Corpus](http://icrc.hitsz.edu.cn/info/1037/1162.htm): need to apply
- DC: ten-class classification, remove stopwords?
  - [x] [THUCNews](https://github.com/gaussic/text-classification-cnn-rnn)