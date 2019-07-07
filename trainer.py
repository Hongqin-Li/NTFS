import torch 

class Trainer():

    def __init__(self, model, optimizer, metrics, dataset, save_path):
        # metrics: function to compute accuracy for early-stop

        self.dataset = dataset

        self.model = model
        self.optimizer = optimizer

        self.metrics = metrics

        self.save_path = save_path


    # FIXME should modify both load and save function when adding other objects to save
    def load_checkpoint(self):
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

    def save_checkpoint(self):
        torch.save({
            'model': self.model.state_dict(),  
            'optimizer': self.optimizer.state_dict(),  
            'best_score': self.best_score,
            'num_trained_samples': self.num_trained_samples
        })


    def train(self, batch_size):

        self.load_checkpoint()


        while True:


            cnt = 0

            self.model.train()
            for batch in self.dataset.trainset(batch_size=batch_size):

                self.model.zero_grad()
                # NOTE Sequence labeling should consider mask !
                loss = self.model.compute_loss(self.model(batch.input), batch.target)
                loss.backward()
                self.optimizer.step()

                cnt += batch_size

                print (f'idx: {cnt}, loss: {loss}')

            self.model.eval()
            score = self.metrics(model, self.dataset.devset(batch_size=batch_size))

            print (f'Epoch: {self.num_trained_samples // self.dataset.num_train_samples}, score: {score}')

            if score > self.best_score:
                self.best_score = score
                self.num_trained_samples += self.dataset.num_train_samples
                self.save_checkpoint()
               
            # TODO still train for several epochs to find a better model
            else:
                print ('Terminate!')
                return 



if __name__ == '__main__':
       
    pass

