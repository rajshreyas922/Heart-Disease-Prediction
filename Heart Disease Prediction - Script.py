import numpy as np # linear algebra
import pandas as pd # data processing
import sklearn.linear_model
import matplotlib.pyplot as plt #visualization
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split #split data
from sklearn.model_selection import GridSearchCV #get best parameters
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
RANDOM_SEED = 690
CV = 10

#Getting the data
data = pd.read_csv("/Users/rajsh/Desktop/CMPUT 466 Mini Project/heart.csv")
data.info()

#Converting to binary
data['Sex'] = np.where(data['Sex'] == 'M', 1, 0)
data['ExerciseAngina'] = np.where(data['ExerciseAngina'] == 'Y', 1, 0)

#create target value and label
y=data.HeartDisease
X=data.drop('HeartDisease', axis=1)

#Get one hot vectors
X=pd.get_dummies(X)

#Train test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = RANDOM_SEED)

#Baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
preds = dummy.predict(X_test)
print("Performance of Dummy Classifier:")
print(classification_report(y_test, preds, zero_division = 0))

#Logistic Regression
logreg = sklearn.linear_model.LogisticRegression(solver='liblinear', random_state = RANDOM_SEED)
param_grid = {'C': [0.1,0.5,1,10], 'penalty': ['l1', 'l2']}
#Cross Validation
lg = GridSearchCV(logreg, param_grid, verbose=True, 
                   scoring="f1", n_jobs=-1, cv=CV)
lg.fit(X_train, y_train)
preds = lg.predict(X_test)
print("Best parameters for Logistic Regression:")
print(lg.best_params_)
print("Performance of Logistic Regression: ")
print(classification_report(y_test, preds))

#MLP
param_grid={
    'learning_rate': ["constant", "adaptive", "invscaling"],
    'hidden_layer_sizes': [(50,50), (50), (100)],
    'alpha': [1e-1, 1e-2],
    'solver': ['adam','sgd'],
    'activation': ["relu", "tanh", "logistic"]
}
#Cross Validation
#Scoring by balanced_accuracy over recall or accuracy gives far better results.
mlp = GridSearchCV(MLPClassifier(random_state = RANDOM_SEED, max_iter = 20000), param_grid, verbose=True, 
                   scoring="balanced_accuracy", n_jobs=-1, cv=CV)

mlp.fit(X_train, y_train)
print("Best parameters for MLP:")
print(mlp.best_params_)
preds = mlp.predict(X_test)
print("Performance of MLP: ")
print(classification_report(y_test, preds))

#SVM
param_grid =[{'kernel': ['poly'], 'degree': [2, 3], 'C': [0.5,0.1,10]},
            {'kernel': ['linear', 'sigmoid'], 'C': [0.5,0.1,10]}]
#Cross Validation
svm = GridSearchCV(SVC(class_weight='balanced', random_state = RANDOM_SEED), param_grid, verbose=True, 
                   scoring="balanced_accuracy", n_jobs=-1, cv=CV)

svm.fit(X_train, np.ravel(y_train))
print("Best parameters are for SVM:")
print(svm.best_params_)
preds = svm.predict(X_test)
print("Performance of SVM: ")
print(classification_report(y_test, preds))