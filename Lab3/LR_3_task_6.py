import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")


# Вхідні дані
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, m)

# Створення об'єкта лінійного регресора
linear_regression = LinearRegression()
linear_regression.fit(X, y)
# Побудова графіка лінійної регресії
plot_learning_curves(linear_regression, X, y)
plt.xticks(())
plt.yticks(())
plt.show()

polynomial_regression = Pipeline(
    [("poly_features", PolynomialFeatures(degree=10, include_bias=False)), ("linear_regression", LinearRegression())])
plot_learning_curves(polynomial_regression, X, y)
plt.show()

polynomial_regression = Pipeline(
    [("poly_features", PolynomialFeatures(degree=2, include_bias=False)), ("linear_regression", LinearRegression())])
plot_learning_curves(polynomial_regression, X, y)
plt.show()
