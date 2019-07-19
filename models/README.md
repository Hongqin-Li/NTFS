# Model

A model should be able to compute loss between inputs and outputs, make prediction given inputs, thus the framework is as below:

```python
def Model():
    def __init__(self):
        pass
    def forward(self, x):
       	pass
    def compute_loss(self, pred, target):
        pass
    def predict(self, x):
        pass
```

