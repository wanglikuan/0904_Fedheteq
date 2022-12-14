import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
import numpy as np
import time
from flcore.clients.byol_pytorch import BYOL

class clientBYOL(Client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.learner = BYOL(
            self.model,
            image_size = 32,
            hidden_layer = 'avgpool'
        )

        self.opt = torch.optim.Adam(self.learner.parameters(), lr=self.learning_rate)

        #print('learner param')
        #for name,parameters in self.learner.named_parameters():
        #    print(name,':',parameters.size())

    def get_next_train_batch_contrastive(self):
        try:
            # Samples a new batch for persionalizing
            ((x_i, x_j), y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            ((x_i, x_j), y) = next(self.iter_trainloader)

        if type(x_i) == type([]):
            x_i[0] = x_i[0].to(self.device)
            x_j[0] = x_j[0].to(self.device)  
        else:
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)  
        y = y.to(self.device)

        return x_i, x_j, y


    def train(self):
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            images_i, images_j, y = self.get_next_train_batch_contrastive()
            loss = self.learner(images_i,images_j)
            #print('training loss got')        
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.learner.update_moving_average() # update moving average of target encoder

        ##################???????????????#############################
        loss_sum = 0.0
        data_num = 0.0
        for step in range(max_local_steps//5):
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            images_i, images_j, y = self.get_next_train_batch_contrastive()
            #self.optimizer.zero_grad()
            self.opt.zero_grad()
            output = self.model(images_i)
            loss = self.loss(output, y)
            loss.backward()
            #self.optimizer.step()
            self.opt.step()  
            loss_sum += loss.item()  
        data_num = (max_local_steps//5)*self.batch_size
        print('client id: ',self.id,'    loss: ',loss_sum/data_num)   
        ###############################################

        # self.model.cpu()
        torch.save(self.model.state_dict(), 'net_client{}.pt'.format(self.id))
   
        #print('self.model param') 
        #for name,parameters in self.model.named_parameters():
        #    print(name,':',parameters.size())
    
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # def train(self):
    #     start_time = time.time()

    #     # self.model.to(self.device)
    #     self.model.train()
        
    #     max_local_steps = self.local_steps
    #     if self.train_slow:
    #         max_local_steps = np.random.randint(1, max_local_steps // 2)

    #     for step in range(max_local_steps):
    #         if self.train_slow:
    #             time.sleep(0.1 * np.abs(np.random.rand()))
    #         x, y = self.get_next_train_batch()
    #         self.optimizer.zero_grad()
    #         output = self.model(x)
    #         loss = self.loss(output, y)
    #         loss.backward()
    #         self.optimizer.step()

    #     # self.model.cpu()

    #     self.train_time_cost['num_rounds'] += 1
    #     self.train_time_cost['total_cost'] += time.time() - start_time
