import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sb

data = pd.read_csv("C:/Users/admin/Downloads/diabetes_.csv")

sb.heatmap(data.isnull())
pp.show()
#print(data)

'''x = data.iloc[:,1].values
y = data.iloc[:,-1].values

pp.plot(x,y)
pp.xlabel("Pregnancies")
pp.ylabel("Status")
pp.show()'''

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(xtrain)

xtrain = scale.transform(xtrain)
xtest = scale.transform(xtest)

scores = {}
score_list = []

for i in range(1,15):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(xtrain,ytrain)
    ypred = classifier.predict(xtest)

    scores[i] = metrics.accuracy_score(ytest,ypred)
    score_list.append(metrics.accuracy_score(ytest,ypred))

#print(scores)
#print(score_list)

'''result = metrics.confusion_matrix(ytest,ypred)
print(result)

clasRep = metrics.classification_report(ytest,ypred)
print(clasRep)

pp.plot(range(1,15),score_list)
pp.xlabel("Neighbors")
pp.ylabel("Accuracy")
pp.title("Accuracy v/s Neighbors")
pp.show()'''

classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(xtrain,ytrain)

ypred = classifier.predict(xtest)

cnf = metrics.confusion_matrix(ytest,ypred)
print(cnf)

clrp = metrics.classification_report(ytest,ypred)
print(clrp)

sb.heatmap(cnf,annot=True,fmt = "g")
pp.show()

print("Accuracy score = ",metrics.accuracy_score(ytest,ypred))
print("Precision score = ",metrics.precision_score(ytest,ypred))

