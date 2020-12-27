# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_loan.csv', sep = ';')
y = dataset.iloc[:, 12].values #loan status

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

dataset = dataset.drop(columns=['Loan_ID'])
dataset = dataset.drop(columns=['Loan_Status'])



dataset = pd.get_dummies(dataset, columns=['Gender', 'Education','Property_Area','Married','Self_Employed'])


X = dataset.iloc[:, 0:17].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaler = sc.fit(X_train)
X_train = scaler.transform(X_train)
X_test = sc.transform(X_test)


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(random_state = 0)
classifier.fit(X_train, y_train)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()


#plot diagram
plt.scatter(dataset['ApplicantIncome'], dataset['LoanAmount'], color = 'red')
plt.title('Applicant Income vs Loan Amount ')
plt.xlabel('applicant income')
plt.ylabel('loan amount')
plt.show()



#plot diagram
plt.scatter(y,dataset['LoanAmount'],  color = 'green')
plt.title('Loan Status vs Loan Amount ')
plt.xlabel('Loan Status')
plt.ylabel('Loan Amount')
plt.show()


#plot diagram
plt.plot(y,color = 'red', label='predictions')
plt.plot(dataset['Dependents'], color = 'blue',label='dependents')
plt.title('Loan Status vs Dependents')
plt.xlabel('Loan Status')
plt.ylabel('Dependents')
plt.legend(loc='lower right')
plt.show()




#plot test set
plt.plot(y_pred, color = 'red', label='predictions')
plt.plot(y_test, color = 'blue',label='real test values')
plt.title('Loan Status (Test set)')
plt.legend(loc='lower right')
plt.show()



#Evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
print(precision_score(y_true=y_test, y_pred=y_pred))
print(recall_score(y_true=y_test, y_pred=y_pred))
print(f1_score(y_true=y_test, y_pred=y_pred))
print(accuracy_score(y_true=y_test, y_pred=y_pred))



