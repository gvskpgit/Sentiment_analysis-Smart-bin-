from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained ML model
with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Global list to store previous comments and predictions
previous_comments = []

@app.route('/')
def home():
    positive_count = sum(1 for c in previous_comments if c['prediction'] == 'Positive')
    negative_count = sum(1 for c in previous_comments if c['prediction'] == 'Negative')
    return render_template('index.html', previous_comments=previous_comments, positive_count=positive_count, negative_count=negative_count)

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    prediction_num = model.predict([comment])[0]
    print(f"Received comment: {comment}")
    print(f"Prediction: {prediction_num}")

    # Quick fix: override prediction to Negative if 'bad' in comment (case-insensitive)
    if 'bad' in comment.lower():
        prediction = "Negative"
    else:
        prediction = "Positive" if prediction_num == 0 else "Negative"

    # Store the comment and prediction in the global list
    previous_comments.append({'comment': comment, 'prediction': prediction})

    positive_count = sum(1 for c in previous_comments if c['prediction'] == 'Positive')
    negative_count = sum(1 for c in previous_comments if c['prediction'] == 'Negative')

    return render_template('index.html', comment=comment, prediction=prediction, previous_comments=previous_comments, positive_count=positive_count, negative_count=negative_count)

if __name__ == '__main__':
    app.run(debug=True)
