import torch 

import ... as model

model = model(....)


t = trainer(model=model)

class Trainer():

    def __init__(self, model, optimizer, criterion, metrics, dataset, save_path):
        # criterion: function to compute loss
        # metrics: function to compute accuracy for early-stop

        self.dataset = dataset

        self.model = model
        self.optimizer = optimizer

        self.criterion = criterion
        self.metrics = metrics

        self.save_path = save_path


    # FIXME should modify both load and save function when adding other objects to save
    def load_checkpoint():
        try:
            cp = torch.load(self.save_path)

            self.model.load_state_dict(cp['model'])
            self.optimizer.load_state_dict(cp['optimizer'])
            self.best_score = cp['best_score']
            self.num_trained_samples = cp['num_trained_samples']
            print ('Load checkpoint successfully!')
        except:
            self.best_score = 0
            self.num_trained_samples = 0
            print ('No checkpoint found!')

    def save_checkpoint():
        torch.save({
            'model': self.model.state_dict(),  
            'optimizer': self.optimizer.state_dict(),  
            'best_score': self.best_score,
            'num_trained_samples': self.num_trained_samples
        })


    # TODO only api here
    def train(self, batch_size):

        self.load_checkpoint()


        while True:

            self.model.train()
            for batch in self.dataset.trainset(batch_size=batch_size):

                self.model.zero_grad()
                loss = self.criterion(self.model(batch.input), batch.target)
                loss.backward()
                self.optimizer.step()

                print (f'idx: {cnt}, loss: {loss}')

            score = 0

            self.model.eval()
            for batch in self.dataset.devset(batch_size=batch_size):
                score += self.metrics(self.model(batch.input), batch.target)

            score /=  len(self.dataset.devset)
            print (f'Epoch: {self.num_trained_samples // cnt}, score: {score}')

            if score > self.best_score:
                self.best_score = score
                self.num_trained_samples += len(self.dataset.trainset)
                self.save_checkpoint()
               
            # TODO still train for several epochs to find a better model
            else:
                print ('Terminate!')
                return 

            
            
            
       

