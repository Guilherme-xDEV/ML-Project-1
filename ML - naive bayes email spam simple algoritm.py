#spam filtering simple algoritm based in naive bayes machine learning method
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

spam_df = pd.read_csv("spam2.csv", encoding='utf-8')

# Check for any null values and handle them
spam_df.dropna(subset=['Message'], inplace=True)

spam_df.groupby('Category').describe
#create a new column called spam with the data, turn spam/ham into numerical data, 
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

#create test/train split
X_train, X_test, Y_train, Y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25) #25/100 of all data is used for training the model

#find word count and store data as matrix. conver words to numbers using a vectorize function 1 or 0
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train.values)
X_train_count.toarray()

#train model
model = MultinomialNB()
model.fit(X_train_count, Y_train)

""" 
email_ham = ["hey wanna meet up for the game?"] #example of working; [0], since this is not spam
email_ham_count = cv.transform(email_ham)
print(model.predict(email_ham_count))

email_spam = ["reward money click"] # Here we got [1], since this is spam
email_spam_count = cv.transform(email_spam)
print(model.predict(email_spam_count))

X_test_count = cv.transform(X_test)
Accuracy = model.score(X_test_count, Y_test)

print(Accuracy)  
"""

def text_classification():
    while True:
        print("spam verifier")
        user_input = input("Enter possible spam text here: ")
        vectorizer = cv.transform([user_input])  # Wrap user_input in a list
        prediction = model.predict(vectorizer)
        print("Prediction:", prediction)
        print("The text has a probability of being spam of: ", model.predict_proba(vectorizer)[:, 1])

text_classification()

