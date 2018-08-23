import matplotlib
matplotlib.use('Agg')
import numpy as np
from utils import *
from ConvNet import *
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


## create directories for results and modelsgh
if not exists("./results/"):
  makedirs("./results/")

if not exists("./results/test_new"):
  makedirs("./results/test_new/")

if not exists("./weights/"):
  makedirs("./weights/")

if not exists("./weights/test_new"):
  makedirs("./weights/test_new/")

class DriveData(data.Dataset):
  def __init__(self, pos_filename, neg_filename, transform=None):
    self.transform = transform
    X_pos = import_seq(pos_filename)
    X_neg = import_seq(neg_filename)
    self.__xs = X_pos + X_neg
    self.__ys = [0] * len(X_pos) + [1] * len(X_neg)

  def __getitem__(self, index):
    return (encode(self.__xs[index], RNA.fold(self.__xs[index])[0], RNA.fold(self.__xs[index])[1]), self.__ys[index])

  def __len__(self):
    return len(self.__xs)

def update_lr(optimizer, lr):    
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_EPOCH = 80
BATCH_SIZE = 64
NUM_CLASSES = 2
LEARNING_RATE = 0.01
TRAIN_SPECIES = 'whole'
TEST_SPECIES = 'new'


WriteFile = open("./results/test_new/%s_test.rst" % TEST_SPECIES ,"w")
rst = []
loss_list = []
accuracy_list = []
model = ConvNet().to(device)
model = model.double()
weights = [4.0, 1.0]
class_weights = torch.DoubleTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
train_dataset = DriveData("./dataset/sequences/%s_pos.fa" % TRAIN_SPECIES, "./dataset/sequences/%s_neg.fa" % TRAIN_SPECIES)
test_dataset = DriveData("./dataset/sequences/%s_pos.fa" % TEST_SPECIES, "./dataset/sequences/%s_neg.fa" % TEST_SPECIES)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)
curr_lr = LEARNING_RATE
for epoch in range(N_EPOCH):
  print(epoch)
  correct = 0
  total = 0
  loss_total = 0
  for i, (seqs, labels) in enumerate(train_loader):
    seqs = seqs.to(device)
    labels = labels.to(device)
    outputs = model(seqs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_total += loss.item()

  loss_list.append(loss_total)
  accuracy_list.append(float(correct) / total)

  _, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  ax1.plot(loss_list)
  ax2.plot(accuracy_list, 'r')
  ax1.set_xlabel("epoch")
  ax1.set_ylabel("training loss")
  ax2.set_ylabel("training accuracy")
  ax1.set_title("training accuracy and loss")
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  plt.savefig("./results/test_new/accuracy_loss_%s.png" % TEST_SPECIES, dpi=300)
  plt.close()

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
      Y_test.extend(labels)
    rst = perfeval(F.softmax(torch.stack(predictions), dim=1).cpu().numpy(), Y_test, verbose=1)
    wrtrst(WriteFile, rst, 0, epoch)

WriteFile.close()
torch.save(model.state_dict(), "./weights/test_new/%s_test.pt" % TEST_SPECIES)
#model.load_state_dict(torch.load("./weights/test_new/%s_test.pt" % TEST_SPECIES))