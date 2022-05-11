# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

filename = 'diabetes.csv'
dataframe = read_csv(filename)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=5, random_state=2, shuffle=True)
model = KNeighborsClassifier(n_neighbors=5)
results = cross_val_score(model, X, Y, cv=kfold)
#print(results.mean())

leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
clf = GridSearchCV(model, hyperparameters, cv=10)
best_model = clf.fit(X,Y)
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
y_pred = best_model.predict(X)
print(accuracy_score(Y, y_pred))

