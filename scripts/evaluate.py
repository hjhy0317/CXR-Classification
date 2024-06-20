import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_classes))

    fpr = {}
    tpr = {}
    roc_auc = {}

    unique_classes = np.unique(y_test)
    for i in range(len(unique_classes)):
        y_test_bin = (y_test == i).astype(int)
        y_pred_bin = y_pred[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_test_bin, y_pred_bin)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(len(unique_classes)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    precision = {}
    recall = {}
    average_precision = {}

    for i in range(len(unique_classes)):
        y_test_bin = (y_test == i).astype(int)
        y_pred_bin = y_pred[:, i]
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin, y_pred_bin)
        average_precision[i] = average_precision_score(y_test_bin, y_pred_bin)

    plt.figure()
    for i in range(len(unique_classes)):
        plt.plot(recall[i], precision[i], label=f'{class_names[i]} (AP = {average_precision[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

    false_negatives = {}
    fn_rate = {}

    for i in range(1, len(unique_classes)):  # Exclude the 'Normal' class
        false_negatives[class_names[i]] = conf_matrix[i][0]  # FN count classified as Normal
        fn_rate[class_names[i]] = conf_matrix[i][0] / sum(conf_matrix[i])  # FN Rate

    print("False Negatives (Disease classified as Normal):")
    for disease, fn_count in false_negatives.items():
        print(f'{disease}: {fn_count}')

    print("\nFalse Negative Rates:")
    for disease, rate in fn_rate.items():
        print(f'{disease}: {rate:.2f}')

    plt.figure(figsize=(10, 5))
    plt.bar(false_negatives.keys(), false_negatives.values(), color='red')
    plt.xlabel('Disease Class')
    plt.ylabel('False Negative Count')
    plt.title('False Negatives (Disease classified as Normal)')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(fn_rate.keys(), fn_rate.values(), color='blue')
    plt.xlabel('Disease Class')
    plt.ylabel('False Negative Rate')
    plt.title('False Negative Rates')
    plt.show()
