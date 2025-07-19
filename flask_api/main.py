import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return comment

# Load model and vectorizer
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    try:
        print("🔗 Trying to load model from MLflow...")
        mlflow.set_tracking_uri("http://ec2-34-201-136-168.compute-1.amazonaws.com:5000/")
        client = MlflowClient()
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Loaded model from MLflow")
    except Exception as e:
        print(f"⚠️ MLflow load failed: {e}")
        print("📦 Falling back to local pickle model...")
        with open('./sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Loaded local pickle model")

    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
        print("✅ Loaded vectorizer")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Welcome to our Flask API"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    try:
        data = request.json
        comments_data = data.get('comments')
        if not comments_data:
            return jsonify({"error": "No comments provided"}), 400

        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        preprocessed_comments = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed_comments)

        # Create DataFrame with feature names
        dense_df = pd.DataFrame(
            transformed.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Predict
        try:
            preds = model.predict(dense_df).tolist()
        except Exception as mlflow_err:
            print(f"⚠️ MLflow prediction failed: {mlflow_err}")
            print("📦 Retrying with local pickle model...")
            with open('./sentiment_model.pkl', 'rb') as f:
                local_model = pickle.load(f)
            preds = local_model.predict(transformed).tolist()

        response = [{"comment": c, "sentiment": str(p), "timestamp": t}
                    for c, p, t in zip(comments, preds, timestamps)]
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [int(sentiment_counts.get('1', 0)),
                 int(sentiment_counts.get('0', 0)),
                 int(sentiment_counts.get('-1', 0))]
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops={'color': 'w'})
        plt.axis('equal')
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Chart generation failed: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed_comments)
        wordcloud = WordCloud(width=800, height=400, background_color='black',
                              colormap='Blues', stopwords=set(stopwords.words('english')),
                              collocations=False).generate(text)
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Wordcloud generation failed: {e}")
        return jsonify({"error": f"Wordcloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]
        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        for sentiment_value in [-1, 0, 1]:
            plt.plot(monthly_percentages.index, monthly_percentages[sentiment_value],
                     marker='o', linestyle='-', label={-1: 'Negative', 0: 'Neutral', 1: 'Positive'}[sentiment_value],
                     color=colors[sentiment_value])
        plt.title('Monthly Sentiment Trends')
        plt.xlabel('Month')
        plt.ylabel('Percentage (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Trend graph generation failed: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
