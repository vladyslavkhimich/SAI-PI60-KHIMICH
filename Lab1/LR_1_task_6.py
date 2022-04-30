import numpy as np
from sklearn import svm
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

input_file = 'Task/data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')

X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = svm.SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(recall_score(y_test, y_pred, average=None))
