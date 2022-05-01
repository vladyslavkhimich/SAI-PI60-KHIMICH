from sklearn import svm
from sklearn.model_selection import train_test_split
from methods import get_X_y, get_result, show_quality_values

X, y, label_encoder = get_X_y()

# Створення SVМ-класифікатора
classifier = svm.SVC(kernel='poly', degree=8, max_iter=50000, random_state=0)
classifier.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier = svm.SVC(kernel='poly', degree=8, max_iter=50000, random_state=0)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

label = get_result(classifier, label_encoder)
print("Label", label)

show_quality_values(classifier, X, y)
