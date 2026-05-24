import pandas as pd
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ==========================================
# STEP 1: LOAD THE DATASET
# ==========================================
data = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=('headers', 'footers', 'quotes'))
docs = pd.Series(data.data[:6])  # Work with 3 documents for speed

# ==========================================
# STEP 2: PREPROCESSING PIPELINE
# ==========================================
# A. Lowercase conversion
docs = docs.str.lower()

# B. Tokenization (Splitting strings into words)
docs = docs.apply(word_tokenize)

# C. Stopword & Punctuation Removal
stop_words = set(stopwords.words('english'))
docs = docs.apply(lambda words: [w for w in words if w.isalpha() and w not in stop_words])

# D. Normalization (Lemmatization)
lemmatizer = WordNetLemmatizer()
docs = docs.apply(lambda words: [lemmatizer.lemmatize(w) for w in words])

# E. Rejoin (Convert list of tokens back to a single string)
docs = docs.apply(" ".join)

print("--- Cleaned Text Pipeline Output ---")
print(docs)

# ==========================================
# STEP 3: TF-IDF VECTORIZATION
# ==========================================
tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(docs)

# Convert sparse matrix to dense pandas DataFrame for visualization
tfidf_df = pd.DataFrame(
    matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

print("\n--- Final TF-IDF Representation ---")
print(tfidf_df.head())