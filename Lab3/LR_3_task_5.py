import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Вхідні дані
m = 100
X = np.linspace(-3, 3, m)
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, m)

X = X.reshape((m, 1))

# Створення об'єкта лінійного регресора
linear_regression = LinearRegression()
linear_regression.fit(X, y)


# Побудова графіка лінійної регресії
plt.scatter(X, y, color='green')
plt.plot(X, linear_regression.predict(X), color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_regression = LinearRegression()
poly_regression.fit(X_poly, y)

print("X[0]", X[0])
print("X_poly", X_poly)
print("Coefficients", poly_regression.coef_)
print("Intercept", poly_regression.intercept_)

# Впорядковуємо точки по осі X
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

# Побудова графіка поліноміальної регресії
plt.scatter(X, y, color='green')
plt.plot(X_grid, poly_regression.predict(poly_features.fit_transform(X_grid)), color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()
