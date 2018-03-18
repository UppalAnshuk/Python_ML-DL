import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
C=10 #controls regularization and margin sizes in SVMs
m=150
svm_clf = Pipeline((("scaler", StandardScaler()),("SGDclassifier", SGDClassifier(loss="hinge", alpha=1/(m*C)),)))
svm_clf.fit(X, y)
