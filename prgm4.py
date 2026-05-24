import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense

data_df = pd.read_csv("stories.csv", sep=',', on_bad_lines='skip')
print("Columns in stories.csv:", data_df.columns)

text_column_name = 'story'

text = " ".join(
    data_df[text_column_name][:20].astype(str)
).lower()

t = Tokenizer()
t.fit_on_texts([text])

seq = t.texts_to_sequences([text])[0]

seqs = pad_sequences([seq[:i+1] for i in range(1,len(seq))])

X,y = seqs[:,:-1], seqs[:,-1]

model = Sequential([
    Embedding(len(t.word_index)+1,50),
    LSTM(50),
    Dense(len(t.word_index)+1,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')

model.fit(X,y,epochs=3)

test = pad_sequences([t.texts_to_sequences(["the spaceship"])[0]],
                      maxlen=X.shape[1])

pred = np.argmax(model.predict(test),axis=1)[0]

print([w for w,i in t.word_index.items() if i==pred][0])