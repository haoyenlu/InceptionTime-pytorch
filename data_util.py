import math
import torch
import numpy as np

def create_dataloader(data,label,batch_size=16,shuffle=True):

  assert data.shape[0] == label.shape[0], "Data size and label size are not the same"
  
  dataset = []
  for s, l in zip(data,label):
    dataset.append([s,l])


  return torch.utils.data.DataLoader(dataset,batch_size,shuffle=shuffle)



