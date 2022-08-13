from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# see source[2]
def heatconmat(y_true, y_pred):
    sns.set_context('talk')
    plt.figure(figsize=(9, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred),
                annot=True,
                fmt='d',
                cbar=False,
                cmap='gist_earth_r',
                yticklabels=sorted(y_true.unique()))
    plt.show()
