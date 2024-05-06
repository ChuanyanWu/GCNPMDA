from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_train, y_train)

baseline = rf.score(X_test, y_test)
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=1, scoring='accuracy')

importances = result.importances_mean

# Visualize permutation importances
plt.bar(range(len(importances)), importances)
plt.xlabel('Feature Index')
plt.ylabel('Permutation Importance')
