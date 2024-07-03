import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(real, prediction,title="Prediction",output_path="./image"):
    fig = plt.figure(figsize = (12,12))
    cm = confusion_matrix(real, prediction)
    f = sns.heatmap(cm, annot=True, fmt='d')
    f.figure.suptitle(title)
    f.figure.savefig(os.path.join(output_path,"Prediction.png"))
    plt.close(fig)


def plot_history(train_loss,test_loss,train_accuracy,test_accuracy,output_path="./images"):
    fig , ax = plt.subplots(nrows=2,figsize=(12,12))
    ax[0].plot(train_loss,label="Train")
    ax[0].plot(test_loss,label="Test")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("losses")
    ax[0].legend()

    ax[1].plot(train_accuracy,label="Train")
    ax[1].plot(test_accuracy,label="Test")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()

    plt.savefig(os.path.join(output_path,"history.png"))
    plt.close(fig)


