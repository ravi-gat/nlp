!pip install gensim
import string
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

df = pd.read_csv("Test.csv")

def preprocess_text(text):
    text = str(text).lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    return text.split()

sentences = df["text"].apply(preprocess_text).tolist()


model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1)

print("Similar to 'movie':", model.wv.most_similar("movie", topn=2))
print("Similar to 'great':", model.wv.most_similar("great", topn=2))


words = ["movie", "film", "story", "director", "actor", "bad", "good", "great"]
vectors = [model.wv[w] for w in words]

pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], color="red", edgecolors="k")

for i, word in enumerate(words):
    plt.annotate(
        word,
        (coords[i, 0], coords[i, 1]),
        xytext=(5, 2),
        textcoords="offset points",
    )

plt.title("Word Embedding Visualisation (IMDB Dataset)")
plt.grid(True)
plt.show()