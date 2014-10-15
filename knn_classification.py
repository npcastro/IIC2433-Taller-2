from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from utils import plot_confusion_matrix
from utils import open_eros_data


def knn_classify(X, y, neighbors=1, test_size=0.3, plot_conf_matrix=True):
    X_train, X_test, y_train, y_test = (
            train_test_split(X, y, random_state=0, test_size=test_size))

    knn_classifier = KNeighborsClassifier(n_neighbors=neighbors,
            algorithm='kd_tree')
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    if plot_conf_matrix:
        title = "KNN Classification with N={0}".format(neighbors)
        plot_confusion_matrix(y_test, y_pred, title)


if __name__ == '__main__':
    n_values = [1, 5, 10, 50]
    X, y = open_eros_data()
    for n in n_values:
        knn_classify(X.as_matrix(), y.tolist(), neighbors=n)

