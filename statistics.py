import numpy as np
from sklearn import metrics
from os.path import exists
from os import makedirs
from Bio import SeqIO  ## fasta read
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAX_LEN = 400 # maximum sequence length
CHANEL = 4


def import_seq(filename):
  seqs = []
  for record in SeqIO.parse(filename, "fasta"):
    a_seq = str(record.seq)
    seqs.append(a_seq)
  return seqs

SPECIES = ["new", "human", "whole"]
classes = ["pos", "neg"]
seq_len = []

for _species in SPECIES:
  
  for category in classes:
    seqs = import_seq("./dataset/sequences/%s_%s.fa" % (_species,category))
    for seq in seqs:
      seq_len.append(len(seq))

plt.figure()
plt.hist(seq_len, bins=50)
plt.axvline(np.mean(seq_len), color='k', linewidth=1)
plt.xlabel("Sequence length")
plt.ylabel("Number of sequences")
plt.savefig('hist.eps', format='eps', dpi=300)
seq_len.sort()
print(seq_len)
print(len(seq_len))