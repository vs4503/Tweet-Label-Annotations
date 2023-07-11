import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('Annotator 2')
    axes.set_xlabel('Annotator 1')
    axes.set_title("Label: " + class_label)


def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)


def get_list(df):
    df_ann = df.drop(["Message"], axis=1).values.tolist()
    for x in range(len(df_ann)):
        df_ann[x] = [int(y) for y in df_ann[x]]
    return df_ann


def main():
    df1 = pd.read_excel("JobQ3b_aj7354.xlsx", "Data to Annotate")
    df1 = df1.drop(["Message Id"], axis=1)
    df2 = pd.read_excel("JobQ3b_vs4503.xlsx", "Data to Annotate")
    df2 = df2.drop(["Message Id"], axis=1)

    df1_lab = df1.replace(np.nan, 0)
    df2_lab = df2.replace(np.nan, 0)

    labels = list(df1_lab.drop(["Message"], axis=1).columns)

    ck_indices = []

    for lab in labels:
        ann1 = df1_lab[lab].tolist()
        ann2 = df2_lab[lab].tolist()
        res = cohen_kappa(ann1, ann2)
        ck_indices.append(res)

    print(ck_indices)

    avg_ck = sum(ck_indices) / len(ck_indices)
    print(avg_ck)

    # Confusion Matrices
    df_ann1 = get_list(df1_lab)
    df_ann2 = get_list(df2_lab)

    y_1 = np.array(df_ann1)
    y_2 = np.array(df_ann2)
    mat = multilabel_confusion_matrix(y_1, y_2)

    fig, ax = plt.subplots(4, 3, figsize=(12, 7))
    for axes, cfs_matrix, label in zip(ax.flatten(), mat, labels):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()