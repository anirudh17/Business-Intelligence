import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import model_selection

g=[]
b=[]
m=[]
a=[]
clf = tree.DecisionTreeClassifier()
data = pd.read_csv("pima-indians-diabetes.csv",sep = "," , header = None)
print data

X = data.values[:, :8]
print X
Y = data.values[:, 8]
print Y


clf = clf.fit(X, Y)

def predi(a):
     b=[]
     for i in a:
           prediction = (clf.predict([i]))
           
           b.append(str(prediction[0]))
     print b
     return b
    

def sort(z):
     for i in z:
         if i == '1.0':
             g.append(i)
         elif i == '0.0':
             m.append(i)
#         elif i == 'b':
#             b.append(i)
     

   
def create(a1,a2,a3):
       tuples=zip(a1,a2,a3)
       for i,j,k in tuples:
            tem = list([i,j,k])
            a.append(tem)         




data2 = pd.read_csv("testing.csv",sep=',',header=None)
a1=data2.values[:,0:]
print data2
for i in a1:
   a.append(list(i))
z=predi(a)
#print z
sort(z)
print g,m


y = [len(g),len(m)]
objects = ('diabetic','nodiabetic')
height = y
y_pos = np.arange(len(objects))

plt.bar(y_pos, height, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xlabel("diabetic or no-diabetic")
plt.ylabel("Number of patient")
plt.title("prediction of diabetic")
plt.show()

kfold = model_selection.KFold(n_splits=5, random_state=1)
results = model_selection.cross_val_score(clf, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0)) 	
