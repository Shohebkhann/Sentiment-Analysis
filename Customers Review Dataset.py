# Natural Language Processing
# Customers Review Dataset
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the dataset
data=pd.read_csv(r"D:\NLP Projects\Customers Review Dataset\Restaurant_Reviews.tsv",delimiter='\t',quoting=3)

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]', ' ',data['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not  word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
x=cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=0)

# Training the Naive Bayes model on the Training set

from sklearn.naive_bayes  import MultinomialNB
classifier=MultinomialNB()
classifier.fit(xtrain,ytrain)

# Predicting the Test set results
ypred=classifier.predict(xtest)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(ytest,ypred)
print(ac)

bias=classifier.score(xtrain,ytrain)
print(bias)

variance=classifier.score(xtest,ytest)
print(variance)

from sklearn.metrics import auc,roc_curve

# Calculate ROC curve
fpr,tpr,thresholds=roc_curve(ytest,ypred)

# Calculate AUC
roc_auc=auc(fpr,tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve For DecisionTreeClassifier')
plt.legend(loc='lower right')
plt.show()