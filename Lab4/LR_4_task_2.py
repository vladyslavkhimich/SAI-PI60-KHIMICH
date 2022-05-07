import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Завантажуємо вхідні дані про іриси
iris = load_iris()
X = iris['data']
y = iris['target']

# Створюємо об'єкт KMeans з параметрами: кількість кластерів - 8, метод ініціалізації - k-means++ (для покращеного
# вибору центроїдів), максимальна кількість ітерацій - 300, значення відносної терпимості - 0.0001, вимкнений режим
# багатослівності, рандом використовується з пакету numpy, вихідні дані не змінюються, автоматичний вибір алгоритму
#
kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None,
                copy_x=True, algorithm='auto')

# Створюємо об'єкт KMeans з параметрами: кількість кластерів - 5
# kmeans = KMeans(n_clusters=5)

# Навчання моделі кластеризації КМеаns
kmeans.fit(X)

# Передбачення вихідних міток
y_kmeans = kmeans.predict(X)

# Графічне відображення точок та центрів кластерів
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# Метод для пошуку кластерів, параметр rseed використовується для ініціалізації рандому
def find_clusters(X, n_clusters, rseed=2):
    # Ініціалізуємо рандом, перемішуємо отримані значення та отримуємо значення центрів
    rnd = np.random.RandomState(rseed)
    i = rnd.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # Знаходимо мінімальні відстані між точками та центрами
        labels = pairwise_distances_argmin(X, centers)

        # отримуємо середні значення масиву
        new_centers = np.array([X[np.array_equal(labels, i)]]).mean(0)

        for i in range(n_clusters):
            # якщо попередньо знайдені центри відповідать поточним - завершуємо цикл
            if np.array_equal(centers, new_centers):
                break
            centers = new_centers

        return centers, labels


# зображаємо нові знайдені точки кластерів при "зерні" рандому 2
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

# зображаємо нові знайдені точки кластерів при "зерні" рандому 0
centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

# знаходимо точки кластерів імпортованим методом та показуємо їх
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

