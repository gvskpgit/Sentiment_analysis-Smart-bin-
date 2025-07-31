import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import string

def preprocess_text(text):
    # Remove punctuation and lowercase the text
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

# Load dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your dataset file

# Apply preprocessing to the text column
df['tweet'] = df['tweet'].apply(preprocess_text)

X = df['tweet']  # Column with text data
y = df['label']  # Column with labels (Positive/Negative)

# Build pipeline: vectorize text + classifier
custom_stop_words = ENGLISH_STOP_WORDS.difference({'well'})

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(lowercase=True, stop_words=custom_stop_words, ngram_range=(1,2))),
    ('classifier', MultinomialNB())
])

# Train model
pipeline.fit(X, y)

# Save model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as 'sentiment_model.pkl'")
