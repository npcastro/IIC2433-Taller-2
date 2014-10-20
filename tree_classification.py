from sklearn import cross_validation
from sklearn import tree

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

from utils import open_eros_data
from utils import plot_confusion_matrix


if __name__ == '__main__':
	
	X, y = open_eros_data()

	num_classes = len(y.unique())

	y_pred_total = []
	y_test_total = []

	k_fold = cross_validation.StratifiedKFold(y, n_folds = 10, indices = True)

	for train_indices, test_indices in k_fold:
		X_train = X.iloc[train_indices]
		y_train = y.iloc[train_indices]

		X_test = X.iloc[test_indices]
		y_test = y.iloc[test_indices]

		clf = tree.DecisionTreeClassifier( criterion = 'entropy')

		# Ajusto el modelo y predigo 
		clf = clf.fit( X_train, y_train )
		y_pred = clf.predict( X_test )

		y_pred_total += y_pred.tolist()
		y_test_total += y_test.tolist()

	precision = precision_score(y_test_total, y_pred_total, average = None)
	recall = recall_score(y_test_total, y_pred_total, average = None)
	f_score = f1_score(y_test_total, y_pred_total, average = None)

	plot_confusion_matrix(y_test_total, y_pred_total, 'Decision Tree Classifier', normed=True)
