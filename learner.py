import torch
import time
import matplotlib.pyplot as plt

class Learner():
    
    def __init__(self, model, train_dataloader, valid_dataloader, optimizer, scheduler, loss_func, device):
        self.model=model
        self.device=device
        self.model=self.model.to(self.device)
        self.train_dataloader=train_dataloader
        self.valid_dataloader=valid_dataloader
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.loss_func=loss_func
        self.accu_train=[]
        self.accu_valid=[]

    def evaluate(self, loader):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        total_items = 0

        with torch.no_grad():  # No gradients needed for evaluation
            for x in loader:
                x = x.to(self.device)
                output, y = self.model(x)
                
                print(y.shape)
                loss = self.loss_func(output, y)
                total_loss += loss.item()
                total_items += 1

        average_loss = total_loss / total_items
        return average_loss


    def fit_one_epoch(self):
        self.model.train()
        losses=[]

        for idx, x in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            x = x.to(self.device)
            output, target = self.model(x)
            output = output.permute(0, 2, 1) # permute to have batch_size, vocab_size, sequence_length
            #print("output of batch", output)
            # count nan values in output
            print("nan values in output", torch.isnan(output).sum())
            loss = self.loss_func(output, target)
            print("loss of batch", loss)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        #accu_train = self.evaluate(self.train_dataloader)
        #accu_valid = self.evaluate(self.valid_dataloader)
        #self.accu_train.append(accu_train)
        #self.accu_valid.append(accu_valid)
        #self.scheduler.step() 
        return sum(losses)/len(losses)

    def fit_epochs(self, number=None):
        for epoch in range(number):
            loss=self.fit_one_epoch()
            print('-' * 59)
            print('| end of epoch {:3d} | loss: {:5.2f}s '.format(
                                        epoch,
                                        loss))
                                           

    def plot_training(self):
        plt.plot(self.accu_train, label="train accuracy")
        plt.plot(self.accu_valid, label="valid accuracy")
        plt.legend()
        plt.show()