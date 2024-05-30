import math
import torch
import numpy as np

def create_dataloader(data,label,batch_size=16,split_size=0.8,type='task'):

  assert data.shape[0] == label.shape[0]

  indices = np.random.permutation(data.shape[0])
  split = math.floor(data.shape[0] * split_size)
  train_index, test_index = indices[:split], indices[split:]
  label = create_label(label,type)
  
  train_dataset = []
  for i in train_index:
    train_dataset.append([data[i],label[i]])

  test_dataset = []
  for i in test_index:
    test_dataset.append([data[i],label[i]])

  return torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True),torch.utils.data.DataLoader(test_dataset,batch_size,shuffle=True)



def create_label(label,type = 'task'):
    assert type in ['task','severity'] , "Can only used 'task' and 'severity' type of label"

    if type == 'task':
       return label[:,:30].astype(float)
    elif type == 'severity':
       return label[:,32:].astype(float)
    
    


