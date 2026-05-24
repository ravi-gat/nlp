import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense

data = pd.read_csv("IMDB Dataset.csv")[:2000]

X = data['review']
y = data['sentiment'].map({'positive':1,'negative':0})

# Tokenize
t = Tokenizer(num_words=5000)
t.fit_on_texts(X)

X = pad_sequences(t.texts_to_sequences(X),maxlen=100)

# Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Model
model = Sequential([
    Embedding(5000,32),
    LSTM(32),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam')

# Train
model.fit(X_train,y_train,epochs=3)

# Predict
pred = (model.predict(X_test)>0.5).astype(int)

print("Accuracy :",accuracy_score(y_test,pred))
print("F1 Score :",f1_score(y_test,pred))