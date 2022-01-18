#Artificial Neural Network

#Importing the librarires
import numpy as np
import tensorflow as tf
import pandas as pd


#Data Preprocessing

#Importing the dataset
dataset = pd.read_csv(r'D:\PYTHON\Project\DATASET\Churn_Modelling.csv')
dataset.head()
dataset.tail()
dataset.columns
dataset.describe()
dataset.dtypes
dataset.info()


dataset.isna().any()

dataset["Exited"].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(dpi=300, figsize=(10,5))
p = sns.heatmap(dataset.corr(), annot=True, cmap='RdYlGn', center=0)
sns.pairplot(dataset)

dataset.hist(figsize=(15,12), bins=15)
plt.title("Features Distribution")
plt.show()


X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values  


#Encoding the categorical data
#When we have categorical data or ordinal data we can go for label encoder
#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:,2])
X

#One Hot Encoding
# when we have nominal data we can go for onehotencoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#We need to mention transformer and remainder
# when we give drop as remainder only the dummy variables will be applied.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

X = X[:, 1:]


#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Building the ANN

#Initialising the ANN
ann = tf.keras.models.Sequential()

#Adding the inputlayer and the first hidden layer
#units is a nueron size
ann.add(tf.keras.layers.Dense(units=6, activation = 'relu'))

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation = 'relu'))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid'))

#Training the ANN

#Compiling the ANN #Cateogorical_crossentropy----multiclass
#In regression optimizer=mse
ann.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['acc'])

#Training the ANN on the training set
model_history = ann.fit(X_train, y_train, batch_size = 1, epochs = 128, validation_split = 0.1)

#Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)


#Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#Model Performance
ann.evaluate(X_test, y_test)


#Visualization
import matplotlib.pyplot as plt

#Accuracy
plt.figure(dpi=300)
plt.plot(model_history.history['acc'])

plt.plot(model_history.history['val_acc'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Training', 'Validation'], loc='lower right')

#Loss
plt.figure(dpi=300)
plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Training', 'Validation'], loc='upper right')