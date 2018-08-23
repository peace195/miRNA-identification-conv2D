import numpy as np
from sklearn import metrics
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from os.path import exists
from os import makedirs, environ
from Bio import SeqIO  ##fasta read
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import sys
sys.path.append(environ['VIENNA_PATH'])
import RNA

MAX_LEN = 400 #maximum sequence length
CHANEL = 4


def import_seq(filename):
  """
  reference: https://github.com/eleventh83/deepMiRGene/blob/master/reproduce/cv.py
  """
  seqs = []
  for record in SeqIO.parse(filename, "fasta"):
    a_seq = str(record.seq)
    seqs.append(a_seq)
  return seqs


def seq2num(a_seq):
  """
  reference: https://github.com/eleventh83/deepMiRGene/blob/master/reproduce/cv.py
  """
  ints_ = [0]*len(a_seq)
  for i, c in enumerate(a_seq.lower()):
    if c == 'c':
      ints_[i] = 1
    elif c == 'g':
      ints_[i] = 2
    elif (c == 'u') | (c == 't'):
      ints_[i] = 3
  return ints_


def find_parentheses(s):
  stack = []
  parentheses_locs = {}
  for i, c in enumerate(s):
    if c == '(':
      stack.append(i)
    elif c == ')':
      try:
        parentheses_locs[stack.pop()] = i
      except IndexError:
        raise IndexError("Too many close parentheses at index {}".format(i))
  if stack:
    raise IndexError("No matching close parenthesis to open parenthesis at index {}".format(stack.pop()))
  return parentheses_locs


def contact_map(a_str, energy):
  cm = [[0 for x in range(len(a_str))] for y in range(len(a_str))]
  parentheses_locs = find_parentheses(a_str)
  for key, value in parentheses_locs.items():
    cm[key][value] = energy
    cm[value][key] = energy
  return cm


def one_hot(X_enc):
  X_enc_len = len(X_enc)
  index_offset = np.arange(X_enc_len) * CHANEL
  #X_enc_vec = np.ones((X_enc_len, CHANEL)) * 0.2
  #X_enc_vec.flat[index_offset + np.asarray(X_enc).ravel()] = 0.8
  X_enc_vec = np.zeros((X_enc_len, CHANEL))
  X_enc_vec.flat[index_offset + np.asarray(X_enc).ravel()] = 1
  return X_enc_vec

def encode(a_seq, a_str, energy, variable_size = False):
  cm = contact_map(a_str, energy)
  one_hot_seq = one_hot(seq2num(a_seq))
  h_enc = np.zeros((CHANEL, len(a_str), len(a_str)))
  v_enc = np.zeros((CHANEL, len(a_str), len(a_str)))
  for i in range(CHANEL):
    h_enc[i] = np.tile(np.array([one_hot_seq[:, i]]).transpose(), (1, len(a_str)))
    v_enc[i] = np.tile(np.array([one_hot_seq[:, i]]), (len(a_str), 1))

  if variable_size:
    return np.concatenate((h_enc, v_enc, np.expand_dims(cm, axis=0)), axis=0)

  hv_end = np.concatenate((h_enc, v_enc, np.expand_dims(cm, axis=0)), axis=0)
  hv_end = np.concatenate((hv_end, np.zeros((1+2*CHANEL, len(a_seq), MAX_LEN - len(a_seq)))), axis=2) 
  hv_end = np.concatenate((hv_end, np.zeros((1+2*CHANEL, MAX_LEN - len(a_seq), MAX_LEN))), axis=1)
  return hv_end


def perfeval(predictions, Y_test, verbose=0):
  """
  reference: https://github.com/eleventh83/deepMiRGene/blob/master/reproduce/cv.py
  """
  class_label = np.uint8(np.argmax(predictions, axis=1))
  R = np.asarray(Y_test)
  R_one_hot = np.array([ [ float(i == label) for i in range(2) ] for label in R ])
  CM = metrics.confusion_matrix(R, class_label, labels=None)
  CM = np.double(CM)
  acc = (CM[0][0]+CM[1][1])/(CM[0][0]+CM[0][1]+CM[1][0]+CM[1][1])
  se = (CM[0][0])/(CM[0][0]+CM[0][1])
  sp = (CM[1][1])/(CM[1][0]+CM[1][1])
  f1 = (2*CM[0][0])/(2*CM[0][0]+CM[0][1]+CM[1][0])
  ppv = (CM[0][0])/(CM[0][0]+CM[1][0])
  mcc = (CM[0][0]*CM[1][1]-CM[0][1]*CM[1][0])/np.sqrt((CM[0][0]+CM[0][1])*(CM[0][0]+CM[1][0])*(CM[0][1]+CM[1][1])*(CM[1][0]+CM[1][1]))
  gmean = np.sqrt(se*sp)
  try:
    auroc = metrics.roc_auc_score(R_one_hot[:, 0], predictions[:, 0])
    aupr = metrics.average_precision_score(R_one_hot[:, 0], predictions[:, 0], average="micro")
  except ValueError:
    auroc = 0
    aupr = 0

  if verbose == 1:
    print("ACC:","{:.3f}".format(acc), "SE:","{:.3f}".format(se),"SP:","{:.3f}".format(sp),"F-Score:","{:.3f}".format(f1), "PPV:","{:.3f}".format(ppv),"gmean:","{:.3f}".format(gmean),"AUROC:","{:.3f}".format(auroc), "AUPR:","{:.3f}".format(aupr))

  return [se,sp,f1,ppv,gmean,auroc,aupr,acc,CM]


def wrtrst(filehandle, rst, nfold=0, nepoch=0):
  """
  reference: https://github.com/eleventh83/deepMiRGene/blob/master/reproduce/cv.py
  """
  filehandle.write(str(nfold+1)+" "+str(nepoch+1)+" ")
  filehandle.write("SE: %s SP: %s F-score: %s PPV: %s g-mean: %s AUROC: %s AUPR: %s ACC: %s\n" %
  ("{:.3f}".format(rst[0]),
  "{:.3f}".format(rst[1]),
  "{:.3f}".format(rst[2]),
  "{:.3f}".format(rst[3]),
  "{:.3f}".format(rst[4]),
  "{:.3f}".format(rst[5]),
  "{:.3f}".format(rst[6]),
  "{:.3f}".format(rst[7])))
  filehandle.flush()
  return


#for enc in encode("ACGUAA", "((.)).", -10):
#  print enc