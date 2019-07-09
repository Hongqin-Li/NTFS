import torch 

class Trainer():

    def __init__(self, model, optimizer, metrics, dataset, save_path, num_save_steps=100):
        # metrics: function to compute accuracy for early-stop

        self.dataset = dataset

        self.num_save_steps = num_save_steps

        self.model = model
        self.optimizer = optimizer

        self.metrics = metrics

        self.save_path = save_path
        self.save_path_tmp = save_path + '.tmp'


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


    def save_checkpoint(self, path):

        print ('Saving model...', end='')

        torch.save({
            'model': self.model.state_dict(),  
            'optimizer': self.optimizer.state_dict(),  
            'best_score': self.best_score,
            'num_trained_samples': self.num_trained_samples
        }, path)

        print ('Finish.')

    def train(self, batch_size):

        self.load_checkpoint()


        step = 0

        stop_after_overfit = 10
        cnt_overfit = 0

        while True:

            cnt = 0
           
            for batch in self.dataset.trainset(batch_size=batch_size, drop_last=True):

                self.model.train()
                self.model.zero_grad()
                # NOTE Sequence labeling should consider mask !
                loss = self.model.compute_loss(self.model(batch.input), batch.target)
                loss.backward()
                self.optimizer.step()

                print (f'idx: {cnt}, loss: {loss}')

                cnt += batch_size
                step += 1

                if step >= self.num_save_steps: 


                    self.model.eval()
                    score = self.metrics(self.model, self.dataset.devset)

                    print (f'score: {score}')

                    if score > self.best_score:

                        self.best_score = score
                        self.num_trained_samples += step * batch_size
                        self.save_checkpoint(self.save_path)
                        cnt_overfit = 0

                    else:
                        cnt_overfit += 1
                        if cnt_overfit >= stop_after_overfit:
                            print ('Terminate!')
                            return 

                    step = 0
            # print (f'Epoch: {self.num_trained_samples // self.dataset.num_train_samples}, score: {score}')




if __name__ == '__main__':
       
    pass

