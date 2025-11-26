from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models at startup
print("Loading models...")
vectorizer = tf.keras.models.load_model("vectorizer.keras")
model = tf.keras.models.load_model("emotion_model.keras")
le = joblib.load("label_encoder.pkl")
print("Models loaded successfully!")

@app.route("/", methods=["GET"])
def health():
    return {"status": "healthy", "service": "emotion-analysis-api"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.json or "text" not in request.json:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = request.json["text"]
        vec = vectorizer([text])
        probs = model.predict(vec, verbose=0)[0]
        idx = np.argmax(probs)
        label = le.inverse_transform([idx])[0]
        return jsonify({"label": label, "confidence": float(probs[idx])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
