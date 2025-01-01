#spam filtering simple algoritm based in naive bayes machine learning method
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

spam_df = pd.read_csv("spam.csv")
spam_df.groupby('Category').describe
#create a new column called spam with the data, turn spam/ham into numerical data, 
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'span' else 0)

#create test/train split
X_train, X_test, Y_train, Y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25) #25/100 of all data is used for training the model

#find word count and store data as matrix. conver words to numbers using a vectorize function 1 or 0
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train.values)
X_train_count.toarray()

#train model
model = MultinomialNB()
model.fit(X_train_count, Y_train)

email_ham = ["hey wanna meet up for the game?"]
email_ham_count = cv.transform(email_ham)
model.predict(email_ham_count)

email_spam = ["reward money click"]
email_spam_count = cv.transform(email_spam)
model.predict(email_spam_count)

X_test_count = cv.transform(X_test)
Accuracy = model.score(X_test_count, Y_test)

print(Accuracy)