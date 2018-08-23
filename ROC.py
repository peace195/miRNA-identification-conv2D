import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np

test = np.array([[0, 1],
       					[0, 1],
       					[1, 0],
       					[1, 0]])

pred = np.array([[ 0.9,  0.1],
					       [ 0.7,  0.3],
					       [ 0.3,  0.7],
					       [ 0.8,  0.2]])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(test[:, 0], pred[:, 0])
    roc_auc[i] = auc(fpr[i], tpr[i])

print(roc_auc_score(test, pred))
plt.figure()
plt.plot(fpr[1], tpr[1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
