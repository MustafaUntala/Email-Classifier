from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

model_path = 'C:/Users/Mustafa/Desktop/spam_classifier/spam_model.pkl'
vectorizer_path = 'C:/Users/Mustafa/Desktop/spam_classifier/vectorizer.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file '{vectorizer_path}' not found.")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    result = 'spam' if prediction == 1 else 'ham'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
