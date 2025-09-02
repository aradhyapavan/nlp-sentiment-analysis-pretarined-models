import spacy
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import csv
import re
import nltk
import os

# Ensure NLTK uses a writable directory; downloads are handled in Dockerfile/app startup
NLTK_DIR = os.environ.get('NLTK_DATA', os.path.join(os.getcwd(), 'nltk_data'))
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)

# Load spaCy medium English model for text preprocessing (NER, POS tagging)
nlp = spacy.load("en_core_web_md")

# Pre-load transformer models once for faster response
models = {
    "default": pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english"),
    "roberta": pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment"),
    "emotion": pipeline('sentiment-analysis', model="j-hartmann/emotion-english-distilroberta-base")
}

# Initialize stemmer
stemmer = PorterStemmer()

import re

def preprocess_text(text):
    """
    Preprocesses the input text by cleaning, normalizing, tokenizing, stemming, lemmatizing, 
    and extracting Named Entities and POS tags.

    Returns:
    - cleaned_text: Text after removing stop words, punctuation, URLs, and emails.
    - removed_text: Text that was removed during cleaning.
    - normalized_text: Lowercased version of cleaned text.
    - tokenized_text: List of tokens (words) from normalized text.
    - stemmed_tokens: List of stemmed tokens.
    - lemmatized_tokens: List of lemmatized tokens.
    - ner: List of named entities found in the original text.
    - pos: List of Part-of-Speech (POS) tags using normalized text.
    """
    
    # Step 1: Clean the text, removing newlines, multiple spaces, and other unwanted characters
    text = re.sub(r'\s+', ' ', text).strip()  # Replaces any form of multiple whitespace (including \r\n) with a single space

    # Step 2: Apply spaCy NLP for further processing
    doc = nlp(text)

    # Step 3: Cleaning: Remove stop words, punctuations, URLs, and emails
    cleaned_text = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct and not token.like_url and not token.like_email])

    # Removed Text: Contains all stop words, punctuations, URLs, and emails that were filtered
    removed_text = " ".join([token.text for token in doc if token.is_stop or token.is_punct or token.like_url or token.like_email])

    # Step 4: Normalization (lowercasing)
    normalized_text = cleaned_text.lower()

    # Step 5: Tokenization
    tokenized_text = word_tokenize(normalized_text)

    # Step 6: POS tagging on the normalized text (so that it's consistent with tokenized/lemmatized text)
    normalized_doc = nlp(" ".join(tokenized_text))
    pos = [(token.text, token.pos_) for token in normalized_doc if token.pos_ != 'SPACE']

    # Convert tokenized text to a clean list (without brackets or quotes)
    tokenized_text_clean = tokenized_text

    # Step 7: Stemming
    stemmed_tokens = [stemmer.stem(word) for word in tokenized_text]

    # Step 8: Lemmatization
    lemmatized_tokens = [token.lemma_ for token in normalized_doc]

    # Step 9: Named Entity Recognition (NER)
    ner = [(ent.text, ent.label_) for ent in doc.ents]

    return cleaned_text, removed_text, normalized_text, tokenized_text_clean, stemmed_tokens, lemmatized_tokens, ner, pos

def analyze_sentiment(text, model_type="default"):
    """
    Analyze the sentiment of the given text using the specified model type.

    Arguments:
    - text: The input text to analyze.
    - model_type: The sentiment model to use ('default', 'roberta', or 'emotion').

    Returns:
    - sentiment: The overall sentiment of the text (e.g., POSITIVE, NEGATIVE).
    - probabilities: The sentiment probabilities or confidence scores for each label.
    """
    
    classifier = models[model_type]
    results = classifier(text)

    if model_type == 'roberta':
        sentiment_mapping = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }
        sentiment = sentiment_mapping[results[0]['label']]
        confidence = results[0]['score']
        probabilities = [0, 0, 0]
        
        if sentiment == "NEGATIVE":
            probabilities = [confidence, 1 - confidence, 0]
        elif sentiment == "NEUTRAL":
            probabilities = [0, confidence, 1 - confidence]
        else:
            probabilities = [0, 1 - confidence, confidence]

    elif model_type == 'emotion':
        emotions = ['ANGER', 'DISGUST', 'FEAR', 'JOY', 'NEUTRAL', 'SADNESS', 'SURPRISE']
        emotion_probs = [0] * len(emotions)
        for res in results:
            emotion_idx = emotions.index(res['label'].upper())
            emotion_probs[emotion_idx] = res['score']
        probabilities = emotion_probs
        sentiment = results[0]['label'].upper()

    else:
        sentiment = results[0]['label'].upper()
        confidence = results[0]['score']
        probabilities = {
            'NEGATIVE': [confidence, 1 - confidence, 0],
            'POSITIVE': [0, 1 - confidence, confidence]
        }.get(sentiment, [0.3, 0.4, 0.3])

    return sentiment, probabilities

def read_file(file):
    """
    Reads the uploaded file and returns its content as a string.
    Supports .txt and .csv file formats.

    Arguments:
    - file: The uploaded file.

    Returns:
    - Content of the file as a single string.
    """
    
    if file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.filename.endswith('.csv'):
        reader = csv.reader(file.read().decode('utf-8').splitlines())
        return ' '.join([' '.join(row) for row in reader])
    else:
        return None
