import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)

df['target'] = iris.target
df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
print(df)

df0 = df[:50]            # setosa
df1 = df[50:100]         # versicolor
df2 = df[100:]           # virginica

X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_test, y_test)
print(knn.score(X_test, y_test))
