from flask import Flask, request, render_template, make_response
from flask_sqlalchemy import SQLAlchemy
from sentiment_model import preprocess_text, analyze_sentiment, read_file
from wordcloud import WordCloud
import os
import nltk

# Ensure NLTK uses a writable directory inside the container
NLTK_DIR = os.environ.get('NLTK_DATA', os.path.join(os.getcwd(), 'nltk_data'))
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)

# Download required NLTK resources to the writable dir (no-op if present)
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, download_dir=NLTK_DIR, quiet=True)
    except Exception:
        pass

app = Flask(__name__, static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define SentimentRecord model
class SentimentRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    cleaned_text = db.Column(db.Text, nullable=False)
    removed_text = db.Column(db.Text, nullable=False)
    normalized_text = db.Column(db.Text, nullable=False)
    tokenized_text = db.Column(db.Text, nullable=False)
    stemmed_text = db.Column(db.Text, nullable=False)
    lemmatized_text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    ner = db.Column(db.Text, nullable=False)
    pos = db.Column(db.Text, nullable=False)

with app.app_context():
    db.create_all()

# Global variables to store the analysis result
analysis_result = {}

@app.route('/')
def home():
    return render_template('index.html', 
                           sentiment=None, 
                           text=None, 
                           file_uploaded=None,
                           model_type='default')

@app.route('/analyze', methods=['POST'])
def analyze():
    global analysis_result  # To store the results globally for the download
    text = request.form.get('text', '').strip()
    file = request.files.get('file')
    model_type = request.form.get('model_type', 'default')

    file_uploaded = False
    if file and file.filename != '':
        text = read_file(file)
        file_uploaded = True

    if not text or len(text.split()) < 4:
        return render_template('index.html', 
                               error='Please provide at least 4 words for analysis.', 
                               text=text,
                               model_type=model_type,
                               file_uploaded=file_uploaded)

    word_count = len(text.split())
    if word_count > 300:
        return render_template('index.html', 
                               error='Input text exceeds the 300-word limit.', 
                               text=text,
                               model_type=model_type,
                               file_uploaded=file_uploaded)

    try:
        # Step 1: Preprocess text (cleaning, normalization, etc.)
        cleaned_text, removed_text, normalized_text, tokenized_text, stemmed_text, lemmatized_text, ner, pos = preprocess_text(text)

        # Step 2: Use lemmatized text for sentiment analysis
        lemmatized_text_joined = " ".join(lemmatized_text)
        sentiment, probabilities = analyze_sentiment(lemmatized_text_joined, model_type=model_type)

        # Word-level sentiment analysis
        neutral_words, positive_words, negative_words = [], [], []

        if model_type != 'emotion':
            for word in lemmatized_text:
                word_sentiment, _ = analyze_sentiment(word, model_type=model_type)
                if word_sentiment == 'POSITIVE':
                    positive_words.append(word)
                elif word_sentiment == 'NEGATIVE':
                    negative_words.append(word)
                elif word_sentiment == 'NEUTRAL':
                    neutral_words.append(word)

            word_sentiment_distribution = {
                'positive': len(positive_words),
                'neutral': len(neutral_words),
                'negative': len(negative_words)
            }
        else:
            # Emotion model word-level sentiment analysis
            emotion_counters = {
                'ANGER': 0, 'DISGUST': 0, 'FEAR': 0, 'JOY': 0, 'NEUTRAL': 0, 'SADNESS': 0, 'SURPRISE': 0
            }
            emotion_words = {
                'ANGER': [], 'DISGUST': [], 'FEAR': [], 'JOY': [], 'NEUTRAL': [], 'SADNESS': [], 'SURPRISE': []
            }
            for word in lemmatized_text:
                word_sentiment, _ = analyze_sentiment(word, model_type=model_type)
                if word_sentiment in emotion_counters:
                    emotion_counters[word_sentiment] += 1
                    emotion_words[word_sentiment].append(word)

            word_sentiment_distribution = {
                'anger': emotion_counters['ANGER'],
                'disgust': emotion_counters['DISGUST'],
                'fear': emotion_counters['FEAR'],
                'joy': emotion_counters['JOY'],
                'neutral': emotion_counters['NEUTRAL'],
                'sadness': emotion_counters['SADNESS'],
                'surprise': emotion_counters['SURPRISE']
            }

        # Store the analysis result in global variable for download
        analysis_result = {
            'sentiment': sentiment,
            'model_type': model_type,
            'cleaned_text': cleaned_text,
            'removed_text': removed_text,
            'normalized_text': normalized_text,
            'tokenized_text': tokenized_text,
            'stemmed_text': stemmed_text,
            'lemmatized_text': lemmatized_text,
            'ner': ner,
            'pos': pos,
            'original_text': text,
            'word_sentiment_distribution': word_sentiment_distribution,
            'positive_words': positive_words,
            'negative_words': negative_words,
            'neutral_words': neutral_words if model_type != 'emotion' else [],
            'emotion_words': emotion_words if model_type == 'emotion' else None
        }

        # Generate Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(lemmatized_text_joined)
        wordcloud_path = os.path.join('static', 'wordcloud.png')
        wordcloud.to_file(wordcloud_path)

        return render_template('index.html', 
                               sentiment=sentiment, 
                               cleaned_text=cleaned_text, 
                               removed_text=removed_text, 
                               normalized_text=normalized_text,
                               tokenized_text=tokenized_text, 
                               stemmed_text=" ".join(stemmed_text), 
                               lemmatized_text=" ".join(lemmatized_text), 
                               ner=ner, 
                               pos=pos,
                               probabilities=probabilities, 
                               wordcloud_url=wordcloud_path,
                               word_sentiment_distribution=word_sentiment_distribution,
                               positive_words=positive_words, 
                               negative_words=negative_words,
                               neutral_words=neutral_words if model_type != 'emotion' else [],
                               emotion_words=emotion_words if model_type == 'emotion' else None,
                               text=text,
                               model_type=model_type,
                               total_words=len(tokenized_text), 
                               file_uploaded=file_uploaded)
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', 
                               error='An error occurred during analysis.', 
                               text=text,
                               model_type=model_type,
                               file_uploaded=file_uploaded)

@app.route('/download')
def download_result():
    global analysis_result
    try:
        if not analysis_result:
            return "No analysis available for download", 400

        # Build content for the TXT file
        content = f"""
Sentiment
Overall Sentiment: {analysis_result['sentiment']}

Model Used
Selected Model: {analysis_result['model_type']}

Original Text:
{analysis_result['original_text']}

Text Preprocessing Results
Cleaned Text:
{analysis_result['cleaned_text']}

Removed Text:
{analysis_result['removed_text']}

Normalized Text:
{analysis_result['normalized_text']}

Tokenized Text:
{', '.join(analysis_result['tokenized_text'])}

Stemmed Text:
{" ".join(analysis_result['stemmed_text'])} 

Lemmatized Text:
{" ".join(analysis_result['lemmatized_text'])}

Named Entities (NER):
{', '.join([f"{entity[0]} ({entity[1]})" for entity in analysis_result['ner']])}

POS Tags:
{', '.join([f"{word} ({tag})" for word, tag in analysis_result['pos']])}

Total Words: {len(analysis_result['tokenized_text'])}

"""
        # If the model is 'emotion', include emotion-based words
        if analysis_result['model_type'] == 'emotion':
            content += "\nEmotion-Specific Words:\n"
            for emotion, words in analysis_result['emotion_words'].items():
                content += f"{emotion.capitalize()} Words: {len(words)}\n"
                content += f"{', '.join(words)}\n"

        # Otherwise, include positive, neutral, and negative words for other models
        else:
            content += f"""
Positive Words: {len(analysis_result['positive_words'])}
{', '.join(analysis_result['positive_words'])}

Neutral Words: {len(analysis_result['neutral_words'])}
{', '.join(analysis_result['neutral_words'])}

Negative Words: {len(analysis_result['negative_words'])}
{', '.join(analysis_result['negative_words'])}
"""

        # Create a response object with the content
        response = make_response(content)
        response.headers["Content-Disposition"] = "attachment; filename=sentiment_analysis_result.txt"
        response.headers["Content-Type"] = "text/plain"
        return response
    except Exception as e:
        print(f"Error during file download: {e}")
        return "Error in generating file", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
