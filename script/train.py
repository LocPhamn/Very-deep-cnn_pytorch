import os

import torch
from scipy.cluster.hierarchy import weighted
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as op
import torch.nn as nn
from torchvision.transforms import ToTensor,Resize,Compose
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from models import MyNeuronNetwork
from sklearn.metrics import accuracy_score , confusion_matrix
from animal_10 import Animal10
import shutil
from PIL import Image
from tqdm.autonotebook import tqdm
import argparse
import matplotlib.pyplot as plt



# NHẬP SIÊU THAM SỐ QUA TERMINAL
def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--epochs","-e", type=int, default=50, help="the number of epochs")
    parse.add_argument("--batch-size","-b", type=int, default=32, help="the number of batch")
    parse.add_argument("--num-workers","-n", type=int, default=8, help="the number of core flow")
    parse.add_argument("--checkpoint-dir","-c", type= str, default="train_model", help="the path of best score and last score")
    parse.add_argument("--tensorboard-dir","-t", type= str, default="tensor_board", help="the path of loss and accuracy report ")
    parse.add_argument("--image-size","-i", type= int, default=224, help="Image Size")
    parse.add_argument("--lr","-l", type= float, default=0.001, help="learning rate")
    parse.add_argument("--momemtum","-m", type= float, default=0.9, help="momemtum")
    parse.add_argument("--early-stop-duration","-d", type= int, default=5, help="early_stop_duration")

    arg = parse.parse_args()
    return arg

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="hot")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def train():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    check_point_dir = args.checkpoint_dir
    tensor_board_dir = args.tensorboard_dir
    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size))
    ])
    train_dataset = Animal10(r"D:\Python plus\AI_For_CV\CV_Project\Deep_cnn_pytorch\dataset\animals", True, transform)
    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size= batch_size,
        shuffle=True,
        num_workers= num_workers,
        drop_last= True
    )
    valid_dataset = Animal10(r"D:\Python plus\AI_For_CV\CV_Project\Deep_cnn_pytorch\dataset\animals", False, transform)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    # model = MyNeuronNetwork(num_class=10)
    model = resnet18(weights = ResNet18_Weights)
    model.fc = nn.Linear(in_features=512, out_features=len(train_dataset.categories), bias=True)
    model.to(device)
    criterian = nn.CrossEntropyLoss()
    # optimizer = op.SGD(model.parameters(),lr=args.lr , momentum=args.momemtum)
    optimizer = op.Adam(model.parameters(),lr=args.lr)
    nums_interation = len(train_loader)
    best_accuracy = -1
    best_epoch = 0

    #THƯ MỤC LUU QUA TRINH TRAIN
    if os.path.isdir(check_point_dir):
        shutil.rmtree(check_point_dir)
    os.mkdir(check_point_dir)

    if not os.path.isdir(tensor_board_dir):
        os.makedirs(tensor_board_dir)
        
    writter = SummaryWriter(log_dir="tensor_board")

    # QUÁ TRÌNH TRAIN
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_accuracy = []
        progess_bar = tqdm(train_loader, colour= "yellow")
        for iter,(image, label) in enumerate(progess_bar):
            #Forward
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss_score = criterian(output, label)
            predict = torch.argmax(output, 1)
            train_loss.append(loss_score.item())
            train_accuracy.append(accuracy_score(label.tolist(),predict.tolist()))

            #Backward
            optimizer.zero_grad()
            loss_score.backward()
            optimizer.step()
            progess_bar.set_description("epochs {}/{} loss: {:0.4f}".format(epoch+1,epochs,loss_score.item()))
            writter.add_scalar("Train/Loss",np.mean(train_loss),global_step=(epoch * nums_interation) + iter)
            writter.add_scalar("Train/accuracy",np.mean(train_accuracy),global_step=(epoch * nums_interation) + iter)

        all_labels = []
        all_predictions = []
        all_loss = []

        #QUÁ TRÌNH VALID
        model.eval()
        with torch.no_grad():
            for iter, (image, label) in enumerate(valid_loader):
                image = image.to(device)
                label = label.to(device)
                # Forward
                output = model(image) #output shape [Batch,Num_class]
                loss_score = criterian(output, label)
                predict = torch.argmax(output,1)
                all_labels.extend(label.tolist())
                all_predictions.extend(predict.tolist())
                all_loss.append(loss_score.item())

            acurracy = accuracy_score(all_labels, all_predictions)
            confus_matrix = confusion_matrix(all_labels, all_predictions)
            plot_confusion_matrix(writter, confus_matrix, train_dataset.categories , epoch)
            avg_loss = np.mean(all_loss)
            writter.add_scalar("Valid/Loss", avg_loss, global_step=epoch)
            writter.add_scalar("Valid/accuracy", acurracy, global_step=epoch)
            check_point = {
                "epoch" : epoch,
                "model": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "best_epoch" : best_epoch,
                "best_accuracy":best_accuracy
            }
            print("epochs {}/{} Loss:{:0.4f}  accuracy: {}".format(epoch+1,epochs,avg_loss,acurracy))

            if acurracy > best_accuracy:
                best_accuracy = acurracy
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best.pt"))
                best_epoch = epoch

        # LƯU BẢN MỚI NHẤT VỀ MO HINH
        torch.save(check_point,os.path.join(args.checkpoint_dir, "last.pt"))
    if epoch - best_epoch > args.early_stop_duration:
        print("stop training process at epoch {}".format(epoch+1))
        exit(0)

if __name__ == '__main__':
    train()




