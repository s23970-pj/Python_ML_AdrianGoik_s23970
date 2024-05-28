import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict

# Dane treningowe
train_data = np.array([
    [1, 7], [1, 6], [1, 4], [2, 3], [2,2],  # setosa
    [4, 6], [5, 5], [5, 4], [6, 3], [6, 4], [8, 2],  # versicolor
    [5, 8], [6, 7], [6, 6], [7, 6], [7,4], [8,5]                 # virginica
])
train_labels = np.array([
    'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
    'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
    'virginica', 'virginica', 'virginica', 'virginica','virginica','virginica'
])

# Dane testowe
test_data = np.array([
    [2.7, 6],
    [5, 7],
    [7, 3.5],
    [9, 3],
    [2, 5]
])

#Klasa(prawdziwe)
true_labels = np.array([
    'versicolor', 'virginica', 'versicolor', 'virginica', 'setosa'
])

#klasyfikatory dla k=1 i k=3

knn_1=KNeighborsClassifier(n_neighbors=1)
knn_1.fit(train_data,train_labels)
knn_3=KNeighborsClassifier(n_neighbors=3)
knn_3.fit(train_data,train_labels)

pred_knn1=knn_1.predict(test_data)
pred_knn3=knn_3.predict(test_data)
# przechowywanie wyników użyłem słownika, który po prostu będę aktualizował
res_k1=defaultdict(lambda: {'correct':0, 'total':0})
res_k3=defaultdict(lambda: {'correct':0, 'total':0})
#wyświetlenie wyników
print("\nwyniki dla klasyfikatora k=1")
for i, test_point in enumerate(test_data):
    print(f"Punkt testowy {test_point} ==> przewidywana klasa: {pred_knn1[i]}, prawdziwa klasa: {true_labels[i]}")
print("\nwyniki dla klasyfikatora k=3")
for i, test_point in enumerate(test_data):
    print(f"Punkt testowy {test_point} ==> przewidywana klasa: {pred_knn3[i]}, prawdziwa klasa: {true_labels[i]}")

# Ewaluacja dla kNN (k=1)
for true_label, pred_label in zip(true_labels, pred_knn1):
    res_k1[true_label]['total'] += 1
    if true_label == pred_label:
        res_k1[true_label]['correct'] += 1

# Ewaluacja dla kNN (k=3)
for true_label, pred_label in zip(true_labels, pred_knn3):
    res_k3[true_label]['total'] += 1
    if true_label == pred_label:
        res_k3[true_label]['correct'] += 1

# Obliczanie dokładności w %
def calculate_accuracy(results):
    accuracy = {}
    for key, value in results.items():
        accuracy[key] = (value['correct'] / value['total']) * 100
    return accuracy

accuracy_knn1 = calculate_accuracy(res_k1)
accuracy_knn3 = calculate_accuracy(res_k3)

print("\nDokładność kNN dla k=1:")
print(accuracy_knn1)

print("\nDokładność kNN dla k=3:")
print(accuracy_knn3)
