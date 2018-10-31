import matplotlib

matplotlib.use('Agg')
from utils import *
from ConvNet import *
import torch
import torch.utils.data as data
from os.path import exists
from os import makedirs, environ
import torch.nn.functional as F
import torch.nn as nn
import sys

sys.path.append(environ['VIENNA_PATH'])
import RNA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

## create directories for results and modelsgh
if not exists("./results/"):
  makedirs("./results/")

if not exists("./results/test"):
  makedirs("./results/test/")

if not exists("./weights/"):
  makedirs("./weights/")

if not exists("./weights/test"):
  makedirs("./weights/test/")


def update_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


class DriveData(data.Dataset):
  def __init__(self, pos_filename, neg_filename, transform=None):
    self.transform = transform
    X_pos = import_seq(pos_filename)
    X_neg = import_seq(neg_filename)
    self.__xs = X_pos + X_neg
    self.__ys = [0] * len(X_pos) + [1] * len(X_neg)

  def __getitem__(self, index):
    return (
    encode(self.__xs[index], RNA.fold(self.__xs[index])[0], RNA.fold(self.__xs[index])[1], True), self.__ys[index])

  def __len__(self):
    return len(self.__xs)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_EPOCH = 40
SPECIES = ['human', 'whole']
BATCH_SIZE = 128
NUM_CLASSES = 2
LEARNING_RATE = 0.001

for _species in SPECIES:
  WriteFile = open("./results/test/%s_test_variable.rst" % _species, "w")
  rst = []
  loss_list = []
  accuracy_list = []
  model = ConvNet_v6().to(device)
  model = model.double()
  weights = [4.0, 1.0]
  class_weights = torch.DoubleTensor(weights).to(device)
  criterion = nn.CrossEntropyLoss(weight=class_weights)
  optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
  train_dataset = DriveData("./dataset/cv/%s/%s_pos_all.fa" % (_species, _species),
                            "./dataset/cv/%s/%s_neg_all.fa" % (_species, _species))
  test_dataset = DriveData("./dataset/test/%s/%s_pos_test.fa" % (_species, _species),
                           "./dataset/test/%s/%s_neg_test.fa" % (_species, _species))
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=8, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=8, shuffle=False)
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
      loss.backward()
      if i % BATCH_SIZE == BATCH_SIZE - 1 or i == len(train_loader) - 1:
        optimizer.step()
        optimizer.zero_grad()

      loss_total += loss.item()

    loss_list.append(loss_total / total)
    accuracy_list.append(float(correct) / total)

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(loss_list, label='Loss')
    lns2 = ax2.plot(accuracy_list, 'r', label='Accuracy')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax2.set_ylabel("Training accuracy")
    ax1.set_title("Training accuracy and loss")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=10)

    plt.savefig("./results/test/accuracy_loss_variable_%s.png" % _species, dpi=150)
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
  torch.save(model.state_dict(), "./weights/test/%s_test_variable.pt" % _species)
  # model.load_state_dict(torch.load("./weights/test/%s_test_variable.pt" % _species))
