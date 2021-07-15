import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from itertools import cycle
from sklearn.metrics import classification_report


def model_evaluation(x_test, y_test, model, class_number, tag, font):
    if tag == "machine_learning":
        pre = model.predict_proba(x_test)
    else:
        pre = model.predict(x_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(class_number):
        fpr[i], tpr[i], thresholds_ = roc_curve(y_test[:, i], pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_number)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_number):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= class_number
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 2
    plt.figure(figsize=(7, 7))
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'pink', 'indigo', 'brown'])
    version = ["3", "4", "5", "6", "7", "8"]
    for i, color, v in zip(range(class_number), colors, version):
        label = 'Solidity 0.{0} (area = {1:0.2f})' ''.format(v, roc_auc[i]) if v != "3" else \
            'Solidity ≤0.{0} (area = {1:0.2f})' ''.format(v, roc_auc[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=label)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=14)
    plt.xlabel('False Positive Rate', font)
    plt.ylabel('True Positive Rate', font)
    plt.legend(loc="lower right", fontsize=12.3)
    plt.show()
    for i in range(len(pre)):
        max_value = max(pre[i])
        for j in range(len(pre[i])):
            if max_value == pre[i][j]:
                pre[i][j] = 1
            else:
                pre[i][j] = 0
    print(classification_report(y_test, pre, digits=4))
    print("AC:", accuracy_score(y_test, pre))

