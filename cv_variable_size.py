import matplotlib
matplotlib.use('Agg')
import numpy as np
from utils import *
from ConvNet import *
from ResNet import *
from sklearn import metrics
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from os.path import exists
from os import makedirs, environ
from Bio import SeqIO  ## fasta read
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import sys
sys.path.append(environ['VIENNA_PATH'])
import RNA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if not exists("./results/"):
  makedirs("./results/")

if not exists("./results/cv"):
  makedirs("./results/cv/")

if not exists("./weights/"):
  makedirs("./weights/")

if not exists("./weights/cv"):
  makedirs("./weights/cv/")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
K_FOLD = 5
N_EPOCH = 40
SPECIES = ['human', 'whole']
BATCH_SIZE = 128
NUM_CLASSES = 2
LEARNING_RATE = 0.001


class DriveData(data.Dataset):
  def __init__(self, pos_filename, neg_filename, transform=None):
    self.transform = transform
    X_pos = import_seq(pos_filename)
    X_neg = import_seq(neg_filename)
    self.__xs = X_pos + X_neg
    self.__ys = [0] * len(X_pos) + [1] * len(X_neg)

  def __getitem__(self, index):
    return (encode(self.__xs[index], RNA.fold(self.__xs[index])[0], RNA.fold(self.__xs[index])[1], True), self.__ys[index])

  def __len__(self):
    return len(self.__xs)

for _species in SPECIES:
  WriteFile = open("./results/cv/%s_cv_variable.rst" % _species ,"w")
  rst = []
  for fold in range(K_FOLD):
    loss_list = []
    accuracy_list = []
    model = ConvNet_v6().to(device)
    model = model.double()
    weights = [4.0, 1.0]
    class_weights = torch.DoubleTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
    train_dataset = DriveData("./dataset/cv/%s/train/%s_pos_train_f%d.fa" % (_species,_species,fold+1),
      "./dataset/cv/%s/train/%s_neg_train_f%d.fa" % (_species,_species,fold+1))
    test_dataset = DriveData("./dataset/cv/%s/val/%s_pos_val_f%d.fa" % (_species,_species,fold+1),
      "./dataset/cv/%s/val/%s_neg_val_f%d.fa" % (_species,_species,fold+1))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=8, shuffle=False)
    total_step = len(train_loader)
    for epoch in range(N_EPOCH):
      correct = 0
      total = 0
      loss_total = 0
      print(epoch)
      for i, (seqs, labels) in enumerate(train_loader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        outputs = model(seqs)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        if i % BATCH_SIZE == BATCH_SIZE - 1 or i == len(train_loader) - 1:
          optimizer.step()
          optimizer.zero_grad()

      # Test the model
      model.eval()
      with torch.no_grad():
        predictions = []
        Y_test = []
        for seqs, labels in test_loader:
          seqs = seqs.to(device)
          labels = labels.to(device)
          outputs = model(seqs)
          predictions.extend(outputs.data)
          _, predicted = torch.max(outputs.data, 1)
          Y_test.extend(labels)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          loss_total += criterion(outputs, labels).item()

        wrtrst(WriteFile, perfeval(F.softmax(torch.stack(predictions), dim=1).cpu().numpy(), Y_test, verbose=1), fold, epoch)
      
      loss_list.append(loss_total)
      accuracy_list.append(float(correct) / total)
                 
      _, ax1 = plt.subplots()
      ax2 = ax1.twinx()
      ax1.plot(loss_list)
      ax2.plot(accuracy_list, 'r')
      ax1.set_xlabel("epoch")
      ax1.set_ylabel("validation loss")
      ax2.set_ylabel("validation accuracy")
      ax1.set_title("validation accuracy and loss")
      ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
      plt.savefig("./results/cv/accuracy_loss_%s_%s.png" %(_species, fold), dpi=300)
      plt.close()

    torch.save(model.state_dict(), "./weights/cv/%s_f%d_variable.pt" % (_species,fold+1))
    model.eval()
    with torch.no_grad():
      predictions = []
      Y_test = []
      for seqs, labels in test_loader:
        seqs = seqs.to(device)
        labels = labels.to(device)
        outputs = model(seqs)
        predictions.extend(outputs.data)
        Y_test.extend(labels)
      rst.append(perfeval(F.softmax(torch.stack(predictions), dim=1).cpu().numpy(), Y_test, verbose=0))
  
  rst_avg = np.mean([f[:-1] for f in rst],axis=0)
  print(rst_avg)
  wrtrst(WriteFile, rst_avg, 0, 0)
  WriteFile.close()

