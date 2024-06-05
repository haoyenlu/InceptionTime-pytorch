import argparse
import yaml
import numpy as np

from model import InceptionTime
from data_util import create_dataloader
from train_util import Trainer

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


def load_data(path,config,aug_path = None,split_size=1.0):
    data = np.load(path,allow_pickle=True).item()


    sequence,label = data['data'], data['label']

    if aug_path is not None:
        aug_data = np.load(aug_path,allow_pickle=True).item()
        aug_sequence,aug_label = aug_data['data'],aug_data['label']
        sequence = np.concatenate((sequence,aug_sequence),axis=0)
        label = np.concatenate((label,aug_label),axis=0)

    B,C,T = sequence.shape


    assert config['dataset']['seq_len'] == T and config['dataset']['feature_size'] == C

    train_loader , test_loader = create_dataloader(sequence,label,
                                                   config['dataset']['batch_size'],config['dataset']['split_size'],config['dataset']['label_type'],split_size=split_size)
    

    return  train_loader, test_loader


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default=None)
    parser.add_argument('--aug_data',type=str,default=None)
    parser.add_argument('--ckpt',type=str,default=None)
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--config',type=str,default=None)
    parser.add_argument('--split',type=float,default=1.0)

    

    args = parser.parse_args()

    return args


def create_heatmap(gt, prediction,title="Prediction"):
    
    cm = confusion_matrix(gt prediction)
    f = sns.heatmap(cm, annot=True, fmt='d')
    f.figure.suptitle(title)
    f.figure.savefig("Prediction.png")


def main():
    args = parse_argument()
    config = load_yaml_config(args.config)

    model = InceptionTime(config['dataset']['batch_size'],config['dataset']['seq_len'],config['dataset']['feature_size'],config['dataset']['label_dim'],
                            inception_filter=config['model']['inception_filter'],fcn_filter=config['model']['fcn_filter'],
                            dropout=config['model']['dropout'],depth=config['model']['depth'],fcn_layers=config['model']['fcn_layers'],
                            kernels=config['model']['kernels'],
                            use_residual=config['model']['use_residual'],use_bottleneck=config['model']['use_bottleneck'],use_embedding=config['model']['use_embedding'])
    


    train_dataloader , test_dataloader = load_data(args.data,config,aug_path=args.aug_data,split_size=args.split)

    print(f"Data Size:{len(train_dataloader)}")
    trainer = Trainer(model,config['dataset']['max_iteration'],config['dataset']['lr'],config['dataset']['save_iteration'],args.ckpt)
    trainer.load(args.step)
    prediction, gt , loss, accuracy = trainer.predict(train_dataloader)
    create_heatmap(gt,prediction)

    print(f"Finish Prediction - Loss:{loss:.4f}, Accuracy:{accuracy:.4f}")



if __name__ == '__main__':
    main()


    

