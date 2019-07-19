# Dataset

A minimal version of Dataset is shown below:

```python
from collections import namedtuple
Batch = nametuple('Batch', 'input target')
# batch = Batch(input=..., target=...)
# batch.input and batch.target will be called in `../trainer.py`
class Dataset():
    def __init__(self):
        '''
        self.num_train_samples = ...
        self.num_dev_samples = ...
        self.num_test_samples = ...
        # for classification tasks
        self.num_classes = ...
        '''
    def trainset(self, batch_size, droplast):
        # ...yield batch
    def devset(self, batch_size, droplast):
        # ...yield batch
    def testset(self, batch_size, droplast):
        # ...yield batch
```

