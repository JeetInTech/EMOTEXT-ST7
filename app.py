# app.py
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import os

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define all 28 emotions matching your HTML
all_emotions = [
    "neutral", "admiration", "approval", "gratitude", "amusement", "curiosity",
    "love", "disapproval", "optimism", "anger", "joy", "confusion", "sadness",
    "disappointment", "realization", "caring", "surprise", "excitement", "disgust",
    "desire", "fear", "remorse", "embarrassment", "nervousness", "relief", "pride",
    "grief", "annoyance"
]

# Load pre-trained model and tokenizer
model_name = "bhadresh-savani/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Map model's output to our 28 emotions (custom mapping based on your needs)
emotion_mapping = {
    "sadness": "sadness",
    "joy": "joy",
    "love": "love",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise"
}

# Additional emotions mapping (customize based on your requirements)
secondary_mapping = {
    "disgust": "disgust",
    "optimism": "optimism",
    "neutral": "neutral"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def predict_emotion(text):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        predicted_idx = torch.argmax(logits, dim=1).item()
        
        # Get model's predicted emotion
        model_emotion = model.config.id2label[predicted_idx]
        
        # Map to our extended emotion set
        emotion = emotion_mapping.get(model_emotion.lower(), "neutral")
        
        # Create full probability distribution (custom implementation)
        prob_dict = {emotion: 0.0 for emotion in all_emotions}
        for idx in range(len(model.config.id2label)):
            model_emo = model.config.id2label[idx]
            our_emo = emotion_mapping.get(model_emo.lower(), None) or secondary_mapping.get(model_emo.lower(), "neutral")
            prob_dict[our_emo] += probabilities[idx]
        
        # Normalize probabilities
        total = sum(prob_dict.values())
        prob_dict = {k: v/total for k, v in prob_dict.items()}
        
        # Get confidence for primary emotion
        confidence = prob_dict[emotion]
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_probabilities": prob_dict
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
            
        if len(text) > 1000:
            return jsonify({"error": "Text too long (max 1000 characters)"}), 400
            
        result = predict_emotion(text)
        if not result:
            return jsonify({"error": "Analysis failed"}), 500
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analyze error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)