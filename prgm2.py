import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 1. CHOOSE EASY CATEGORIES (Sports vs Politics)
categories = ['rec.sport.baseball', 'talk.politics.misc']

# 2. LOAD DATA (10 train files, 10 test files)
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

X_train, y_train = train.data[:10], train.target[:10]
X_test, y_test = test.data[:10], test.target[:10]

# 3. VECTORIZE (Lowercasing and stop words are done automatically here!)
tfidf = TfidfVectorizer(lowercase=True, stop_words='english')

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 4. TRAIN AND PREDICT
model = MultinomialNB()
model.fit(X_train_vec, y_train)
preds = model.predict(X_test_vec)

# 5. GENERATE LOTS OF OUTPUT (Using 1 built-in function!)
print("--- Class Names ---")
print("0 =", categories[0])
print("1 =", categories[1])

print("\n--- True Labels vs Predicted Labels ---")
print("Actual Target Labels: ", list(y_test))
print("Model Predict Labels: ", list(preds))

print("\n--- Full Performance Metrics Matrix ---")
print(classification_report(y_test, preds, target_names=categories))