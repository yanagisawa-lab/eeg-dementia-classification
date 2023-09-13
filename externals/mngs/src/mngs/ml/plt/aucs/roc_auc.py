#!/usr/bin/env python3

from itertools import cycle

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def roc_auc(plt, true_class, pred_proba, labels):
    """
    Calculates ROC-AUC curve.
    Return: fig, metrics (dict)
    """

    ## One-hot encoding
    def to_onehot(labels, n_classes):
        eye = np.eye(n_classes, dtype=int)
        return eye[labels]

    # Use label_binarize to be multi-label like settings
    n_classes = len(labels)
    true_class = to_onehot(true_class, n_classes)

    # For each class
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()  # fixme; auc
    for i in range(n_classes):
        try:
            fpr[i], tpr[i], threshold[i] = roc_curve(true_class[:, i], pred_proba[:, i])
            roc_auc[i] = roc_auc_score(true_class[:, i], pred_proba[:, i])            
        except Exception as e:
            print(e)
            roc_auc[i] = 0
            # import ipdb; ipdb.set_trace()


    ################################################################################
    ## Average fpr: micro and macro
    ################################################################################
    # A "micro-average": quantifying score on all classes jointly
    try:
        fpr["micro"], tpr["micro"], threshold["micro"] = roc_curve(
            true_class.ravel(), pred_proba.ravel()
        )
        roc_auc["micro"] = roc_auc_score(true_class, pred_proba, average="micro")
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()
        # Input contains NaN.
        
    # print(
    #     "Average fpr score, micro-averaged over all classes: {0:0.2f}".format(
    #         roc_auc["micro"]
    #     )
    # )

    # macro
    roc_auc["macro"] = roc_auc_score(true_class, pred_proba, average="macro")
    # print(
    #     "Average fpr score, macro-averaged over all classes: {0:0.2f}".format(
    #         roc_auc["macro"]
    #     )
    # )

    ################################################################################
    ## Plot
    ################################################################################
    # Plot Fpr-Tpr curve for each class and iso-f1 curves
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    lines = []
    legends = []

    ## Chance Level (the diagonal line)
    (l,) = ax.plot(
        np.linspace(0.01, 1),
        np.linspace(0.01, 1),
        color="gray",
        lw=2,
        linestyle="--",
        alpha=0.8,
    )
    lines.append(l)
    legends.append("Chance")

    ## Each Class
    for i, color in zip(range(n_classes), colors):
        (l,) = plt.plot(fpr[i], tpr[i], color=color, lw=2)
        lines.append(l)
        legends.append("{0} (AUC = {1:0.2f})" "".format(labels[i], roc_auc[i]))

    # fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend(lines, legends, loc="lower right")

    metrics = dict(roc_auc=roc_auc, fpr=fpr, tpr=tpr, threshold=threshold)

    # return fig, roc_auc, fpr, tpr, threshold
    return fig, metrics


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import softmax
    from sklearn import datasets, svm
    from sklearn.model_selection import train_test_split

    def mk_demo_data(n_classes=2, batch_size=16):
        labels = ["cls{}".format(i_cls) for i_cls in range(n_classes)]
        true_class = np.random.randint(0, n_classes, size=(batch_size,))
        pred_proba = softmax(np.random.rand(batch_size, n_classes), axis=-1)
        pred_class = np.argmax(pred_proba, axis=-1)
        return labels, true_class, pred_proba, pred_class

    ## Fix seed
    np.random.seed(42)

    """
    ################################################################################
    ## A Minimal Example
    ################################################################################    
    labels, true_class, pred_proba, pred_class = \
        mk_demo_data(n_classes=10, batch_size=256)

    roc_auc, fpr, tpr, threshold = \
        calc_roc_auc(true_class, pred_proba, labels, plot=False)
    """

    ################################################################################
    ## MNIST
    ################################################################################
    from sklearn import datasets, metrics, svm
    from sklearn.model_selection import train_test_split

    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001, probability=True)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_proba = clf.predict_proba(X_test)
    predicted = clf.predict(X_test)

    n_classes = len(np.unique(digits.target))
    labels = ["Class {}".format(i) for i in range(n_classes)]

    ## Configures matplotlib
    plt.rcParams["font.size"] = 20
    plt.rcParams["legend.fontsize"] = "xx-small"
    plt.rcParams["figure.figsize"] = (16 * 1.2, 9 * 1.2)

    ## Main
    fig, metrics_dict = roc_auc(plt, y_test, predicted_proba, labels)

    fig.show()

    print(metrics_dict.keys())
    # dict_keys(['roc_auc', 'fpr', 'tpr', 'threshold'])
