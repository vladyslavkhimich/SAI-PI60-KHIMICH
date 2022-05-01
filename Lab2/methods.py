import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score


def get_X_y():
    # Вхідний файл, який містить дані
    input_file = 'Task/income_data.txt'

    # Читання даних
    X = []
    y = []
    count_class1 = 0
    count_class2 = 0
    max_datapoints = 25000

    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break
            if '?' in line:
                continue

            data = line[:-1].split(', ')

            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X.append(data)
                count_class1 += 1
            if data[-1] == '>50K' and count_class2 < max_datapoints:
                X.append(data)
                count_class2 += 1

    # Перетворення на масив numpy
    X = np.array(X)

    # Перетворення рядкових даних на числові
    label_encoder = []
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

    scaler = MinMaxScaler()
    X_encoded = scaler.fit_transform(X_encoded)

    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    return X, y, label_encoder


def get_result(classifier, label_encoder):
    # Передбачення результату для тестової точки даних
    input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family',
                  'White', 'Male', '0', '0', '40', 'United-States']

    # Кодування тестової точки даних
    input_data_encoded = [-1] * len(input_data)
    count = 0
    for i, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[i] = int(input_data[i])
        else:
            input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
            count += 1

    input_data_encoded = np.array(input_data_encoded)

    # Використання класифікатора для кодованої точки даних та виведення результату
    predicted_class = classifier.predict(input_data_encoded.reshape(1, 14))
    return label_encoder[-1].inverse_transform(predicted_class)[0]


def show_quality_values(classifier, X, y):
    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
    print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")
    accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
    print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
    precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=3)
    print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
    recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=3)
    print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
