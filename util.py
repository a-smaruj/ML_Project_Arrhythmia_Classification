import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score


def accuracy_assessment(param_y_test, param_y_pred):
    accuracy = accuracy_score(param_y_test, param_y_pred)
    print("Accuracy:", accuracy)

    balanced_accuracy = balanced_accuracy_score(param_y_test, param_y_pred)
    print("Balanced accuracy:", balanced_accuracy)

    precision = precision_score(param_y_test, param_y_pred, average="weighted")
    print("Precision:", precision)

    recall = recall_score(param_y_test, param_y_pred, average="weighted")
    print("Sensivity (recall):", recall)

    f1 = f1_score(param_y_test, param_y_pred, average="weighted")
    print("F1-Score:", f1)


def our_roc_curves(param_x_test, param_y_test, best_model, title):
    class_labels = ['F', 'N', 'Q', 'SVEB', 'VEB']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    y_pred_proba = best_model.predict_proba(param_x_test)

    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    y_test_bin = label_binarize(param_y_test, classes=np.unique(param_y_test))

    for i, class_label in enumerate(class_labels):
        y_test_bin_i = y_test_bin[:, i]
        y_pred_prob_i = y_pred_proba[:, i]

        fpr, tpr, _ = roc_curve(y_test_bin_i, y_pred_prob_i)
        roc_auc = auc(fpr, tpr)

        fpr_dict[class_label] = fpr
        tpr_dict[class_label] = tpr
        roc_auc_dict[class_label] = roc_auc

        # Przypisanie koloru z listy dostępnych kolorów
        color = colors[i % len(colors)]

        # Ustaw etykiety na legendzie zamiast oryginalnych klas
        plt.plot(fpr, tpr, lw=2, label=f' {class_label} (AUC = {roc_auc:.2f})', color=color)

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', lw=0.8)
    plt.title(title)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right")
    plt.show()


def confussion_matrix(param_y_test, param_y_pred):
    types=np.unique(param_y_test.values)
    conf_matrix=pd.DataFrame(confusion_matrix(param_y_test, param_y_pred, labels=types), columns=types, index=types)
    percentage=conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
    percentage = percentage.fillna(0)
    plt.figure(figsize=(6, 4))
    sns.heatmap(percentage, fmt='.2%', annot=True, annot_kws={"size": 12})
    plt.xlabel("Predicted label")
    plt.ylabel("True label")


def significant_variable(model, param_x_train):

    indeksy = np.where(model.feature_importances_!=0)[0]
    variables= [param_x_train.columns[i] for i in indeksy]
    importances = model.feature_importances_[indeksy]

    # sortowanie
    importances, variables= zip(*sorted(zip(importances, variables), reverse=False))

    # Tworzenie wykresu słupkowego z niezerowymi wartościami
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.barh(variables[:15], importances[:15])

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


def train_test_plot(param_ma_train, param_ma_test, title, x_label, param):
    plt.plot(param, param_ma_train, label="Train Accuracy", color="blue")
    plt.plot(param, param_ma_test, label="Test Accuracy", color="red")
    plt.xlabel(x_label)
    plt.ylabel("Accuracy Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("images/"+ title.lower().replace(" ", "_") + ".png")
    plt.show()
