import torch
import torch.nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import numpy as np


class Trainer:
  def __init__(self,model,max_iteration=1000,lr=0.001,save_iteration = 100, save_path="./save"):
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.model = model.to(self.device)
    self.max_iterations = max_iteration
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr,betas=[0.9,0.99])
    self.loss_fn = torch.nn.CrossEntropyLoss()

    self.save_iteration = save_iteration
    self.save_path = save_path
    os.makedirs(save_path,exist_ok=True)

    self.history = {'train_loss':[],'test_loss':[],'train_accuracy':[],'test_accuracy':[]}

  def fit(self,train_dataloader,test_dataloader):
    step = 0
    print("Start Training")

    with tqdm(initial=step,total=self.max_iterations) as pbar:
      while step < self.max_iterations:
        train_total_loss = 0
        train_total_accuracy = 0

        gt_label = np.array([])
        pred_label = np.array([])
        self.model.train()
        for sequence,label in train_dataloader:
          sequence , label = sequence.float().to(self.device), label.float().to(self.device)
          pred = self.model(sequence)
          train_loss = self.loss_fn(pred,label)
          train_loss.backward()
          self.optimizer.step()
          self.optimizer.zero_grad()
          train_total_loss += train_loss.item()
          train_total_accuracy += (torch.max(label,1)[1] == torch.max(pred,1)[1]).sum().item()

        test_total_loss = 0
        test_total_accuracy = 0

        self.model.eval()
        for sequence,label in test_dataloader:
          sequence , label = sequence.float().to(self.device), label.float().to(self.device)
          pred = self.model(sequence)
          test_loss = self.loss_fn(pred,label)
          test_total_loss += test_loss.item()
          test_total_accuracy += (torch.max(label,1)[1] == torch.max(pred,1)[1]).sum().item()

        step += 1

        if step != 0 and step % self.save_iteration == 0:
          with torch.no_grad():
            self.save(step)

        self.history['train_loss'].append(train_total_loss / len(train_dataloader))
        self.history['test_loss'].append(test_total_loss / len(test_dataloader))
        self.history['train_accuracy'].append(train_total_accuracy / len(train_dataloader))
        self.history['test_accuracy'].append(test_total_accuracy / len(test_dataloader))

        pbar.set_description(f'Train loss: {train_total_loss / len(train_dataloader):.6f}, Train Accuracy: {train_total_accuracy / len(train_dataloader):.2f}%, Test loss: {test_total_loss / len(test_dataloader)}, Test Accuracy: {test_total_accuracy / len(test_dataloader):.2f}%')

        pbar.update(1)

    print("Finish Training")

  def save(self,step):
    torch.save({
        'model': self.model.state_dict(),
        'opt': self.optimizer.state_dict()
    },os.path.join(self.save_path,f"checkpoint-{step}.pt"))

  def load(self,step):
    device = self.device
    data = torch.load(os.path.join(self.save_path,f"checkpoint-{step}.pt"),map_location=device)
    self.model.load_state_dict(data['model'])
    self.optimizer.load_state_dict(data['opt'])

  def predict(self,dataloader):
    test_total_loss = 0
    test_total_accuracy = 0
    gt = np.array([])
    prediction = np.array([])
    self.model.eval()
    for sequence,label in tqdm(dataloader):
      sequence , label = sequence.float().to(self.device), label.float().to(self.device)
      pred = self.model(sequence)
      test_loss = self.loss_fn(pred,label)
      test_total_loss += test_loss.item()
      label_decode = torch.max(label,1)[1]
      pred_decode = torch.max(pred,1)[1]
      prediction = np.append(prediction,pred_decode.detach().cpu().numpy())
      gt = np.append(gt,label_decode.detach().cpu().numpy())
    
    return prediction,gt
  
