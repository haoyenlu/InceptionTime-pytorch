import argparse
import yaml
import numpy as np
import os
import csv

from model import InceptionTime
from data_util import create_dataloader
from train_util import Trainer

from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


from plotting import plot_confusion_matrix, plot_history

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


def load_data(train_path,test_path,aug_path = None,config=None):
    train_data = np.load(train_path,allow_pickle=True).item()
    test_data = np.load(test_path,allow_pickle=True).item()



    if aug_path is not None:
        aug_data = np.load(aug_path,allow_pickle=True).item()
        train_data['data'] = np.concatenate((train_data['data'],aug_data['data']),axis=0)
        train_data['label'] = np.concatenate((train_data['label'],aug_data['label']),axis=0)



    assert config['dataset']['seq_len'] == train_data['data'].shape[2] and config['dataset']['feature_size'] == train_data['data'].shape[1]
    assert config['dataset']['seq_len'] == test_data['data'].shape[2] and config['dataset']['feature_size'] == test_data['data'].shape[1] 
   
    split_size = round(train_data['data'].shape[0] * config['dataset']['split_size'])
    label = np.arange(train_data['data'].shape[0])
    np.random.shuffle(label)
    train_label = label[:split_size]
    valid_label = label[split_size:]

    train_loader = create_dataloader(train_data['data'][train_label],train_data['label'][train_label],config['dataset']['batch_size'])
    valid_loader = create_dataloader(train_data['data'][valid_label],train_data['label'][valid_label],config['dataset']['batch_size'])
    test_loader = create_dataloader(test_data['data'],test_data['label'],config['dataset']['batch_size'])
    

    return  train_loader, valid_loader, test_loader


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',type=str,default=None)
    parser.add_argument('--test_data',type=str,default=None)
    parser.add_argument('--aug_data',type=str,default=None)
    parser.add_argument('--ckpt',type=str,default=None)
    parser.add_argument('--step',type=int,default=None)
    parser.add_argument('--config',type=str,default=None)
    parser.add_argument('--train',action="store_true")
    parser.add_argument('--test',action="store_true")
    parser.add_argument('--plot',action="store_true")
    parser.add_argument('--image_path',type=str,default='./images')
    parser.add_argument('--csv',type=str,default=None)
    

    args = parser.parse_args()

    return args



def main():
    args = parse_argument()
    config = load_yaml_config(args.config)

    model = InceptionTime(config['dataset']['batch_size'],config['dataset']['seq_len'],config['dataset']['feature_size'],config['dataset']['label_dim'],
                          **config.get('model',dict()))
    


    train_dataloader ,valid_dataloader, test_dataloader = load_data(args.train_data,args.test_data,args.aug_data,config=config)

    trainer = Trainer(model,config['dataset']['max_iteration'],config['dataset']['lr'],config['dataset']['save_iteration'],args.ckpt)

    if args.step is not None:
        trainer.load(args.step)

    if args.train:
        trainer.fit(train_dataloader,valid_dataloader)
    
    if args.test:
        prediction, gt = trainer.predict(test_dataloader)
        accuracy = accuracy_score(gt,prediction,normalize=True)
        print(f"Finish Prediction -, Accuracy Score:{accuracy * 100}%")
        
        if args.plot:
            output_path = os.path.join(args.image_path,trainer.id)
            os.makedirs(output_path,exist_ok=True)

            plot_confusion_matrix(real=gt,prediction=prediction,output_path=output_path)
            plot_history(trainer.history['train_loss'],trainer.history['test_loss'],trainer.history['train_accuracy'],trainer.history['test_accuracy'],output_path)
         
        if args.csv is not None:
            fields = config['csv_fields']
            # Insert key to dict
            data = {'train_data':args.train_data,
                    'aug_data':args.aug_data,
                    'test_data':args.test_data,
                    'epoch':config['dataset']['max_iteration'],
                    'accuracy':accuracy}

            if os.path.isfile(args.csv):
                with open(args.csv,'a') as file:
                    writer = csv.DictWriter(file,fieldnames=fields)
                    writer.writerow(data)
            else:
                with open(args.csv,'w') as file:
                    writer = csv.DictWriter(file,fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(data)


if __name__ == '__main__':
    main()


    

