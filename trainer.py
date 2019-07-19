import torch 

class Trainer():

    def __init__(self, model, optimizer, metrics, dataset, save_path, num_save_steps):
        # metrics: function to compute accuracy for early-stop

        self.dataset = dataset


        self.model = model
        self.optimizer = optimizer

        self.metrics = metrics

        self.num_save_steps = num_save_steps
        self.save_path = save_path
        self.save_path_tmp = save_path + '.tmp'


    # NOTE should modify both load and save functions
    def load_checkpoint(self):
        try:
            cp = torch.load(self.save_path)

            self.num_trained_samples = cp['num_trained_samples']
            self.best_score = cp['best_score']
            self.optimizer.load_state_dict(cp['optimizer'])
            self.model.load_state_dict(cp['model'])

            print ('Load checkpoint successfully!')
            print (f'[Best score] {self.best_score}')
            print (f'[Trained samples] {self.num_trained_samples}')
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

    def train(self, batch_size, mini_batch_size=None):

        if mini_batch_size is None: mini_batch_size = batch_size

        try:
            assert batch_size % mini_batch_size == 0
        except: 
            print ('Please set batch size to a multiple of mini-batch size to enable accumulated gradient!')
            raise
        
        accumulation_steps = batch_size // mini_batch_size

        self.load_checkpoint()
        old_num_trained_samples = self.num_trained_samples

        step = 0

        if self.num_save_steps is None:
            num_save_steps = int(0.1 * self.dataset.num_train_samples / batch_size)
        else:
            num_save_steps = self.num_save_steps
        print (f'number of save steps: {num_save_steps}')

        stop_after_overfit = 10
        cnt_overfit = 0

        while True:

            self.model.zero_grad()
            loss_sum = 0

            for i, batch in enumerate(self.dataset.trainset(batch_size=mini_batch_size, drop_last=True)):
                
                self.model.train()

                # NOTE Sequence labeling should consider mask !
                loss = self.model.compute_loss(self.model(batch.input), batch.target) / accumulation_steps
                loss.backward()
                loss_sum += loss

                if (i + 1) % accumulation_steps != 0:
                    continue

                self.optimizer.step()
                self.model.zero_grad()
                step += 1
                print (f'idx: {(i+1) * mini_batch_size}, loss: {loss_sum}')
                loss_sum = 0

                if step % num_save_steps == 0: 

                    self.model.eval()
                    with torch.no_grad():
                        score = self.metrics(self.model, self.dataset.devset, batch_size=mini_batch_size)

                    print (f'score: {score}')

                    if score > self.best_score:
                        print (f'[Best score] {score}')

                        self.best_score = score
                        self.num_trained_samples = old_num_trained_samples + step * batch_size
                        self.save_checkpoint(self.save_path)
                        cnt_overfit = 0

                    else:
                        print (f'[Best score] {self.best_score}')

                        cnt_overfit += 1
                        if cnt_overfit >= stop_after_overfit:
                            print ('Terminate!')
                            print (f'[Trained samples] {self.num_trained_samples}')
                            return 

if __name__ == '__main__':
       
    pass

