# Feature Importance with Extra Trees Classifier
#https://machinelearningmastery.com/feature-selection-machine-learning-python/
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print(names)
print(model.feature_importances_)
