from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np
import os
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Enable CORS for browser access

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load models at startup
print("Loading models...")
model = tf.keras.models.load_model("emotion_model.keras")

# Load vectorizer and extract the TextVectorization layer
print("Loading vectorizer...")
vectorizer_model = tf.keras.models.load_model("vectorizer.keras")
vectorizer = vectorizer_model.layers[0]  # Extract TextVectorization layer
print("Vectorizer loaded successfully!")

# Load label encoder
le = joblib.load("label_encoder.pkl")
print("Models loaded successfully!")

# Map emotions to coaching styles
EMOTION_TO_COACH = {
    'fear': 'CBT coach',  # Anxiety/fear → CBT
    'anger': 'DBT specialist',
    'sadness': 'empathy-focused supporter',
    'disgust': 'DBT specialist',  # Similar to anger
    'neutral': 'productivity therapist',  # Stress often shows as neutral
    'happiness': 'empathy-focused supporter',  # Supportive response
    'love': 'empathy-focused supporter'
}

# Coach-specific system prompts
COACH_PROMPTS = {
    'CBT coach': """You are a Cognitive Behavioral Therapy (CBT) coach specializing in anxiety and fear management. 
    Help users identify and challenge anxious thoughts, provide practical coping strategies, and guide them through 
    evidence-based techniques. Be supportive, practical, and focus on actionable steps.""",
    
    'DBT specialist': """You are a Dialectical Behavior Therapy (DBT) specialist focusing on anger management and 
    emotional regulation. Help users understand their emotions, develop distress tolerance skills, and practice 
    mindfulness techniques. Be calm, non-judgmental, and provide concrete tools for emotional regulation.""",
    
    'empathy-focused supporter': """You are an empathetic mental health supporter specializing in sadness, loneliness, 
    and emotional support. Provide genuine empathy, validate feelings, and offer compassionate guidance. Focus on 
    connection, understanding, and helping users feel heard and supported.""",
    
    'productivity therapist': """You are a productivity therapist specializing in stress management and goal-oriented 
    support. Help users break down overwhelming situations, develop practical action plans, and manage stress through 
    organization and time management techniques. Be solution-focused and encouraging."""
}

def get_llm_response(user_text, emotion, coach_type):
    """Get LLM response based on emotion classification"""
    system_prompt = COACH_PROMPTS.get(coach_type, COACH_PROMPTS['empathy-focused supporter'])
    
    user_prompt = f"""The user shared: "{user_text}"

Their detected emotion is: {emotion}

Please respond as a {coach_type}. Provide a helpful, supportive response that:
1. Acknowledges their emotion
2. Offers appropriate guidance based on your specialization
3. Is concise (2-3 sentences)
4. Is warm and professional

Response:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" for cheaper option
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I'm here to help, but I'm experiencing a technical issue. Please try again. ({str(e)})"

@app.route("/", methods=["GET"])
def health():
    return {"status": "healthy", "service": "emotion-analysis-api"}

@app.route("/predict", methods=["POST"])
def predict():
    """Original endpoint - just returns emotion classification"""
    try:
        if not request.json or "text" not in request.json:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = request.json["text"]
        if isinstance(text, str):
            text = [text]
        
        vec = vectorizer(text)
        probs = model.predict(vec, verbose=0)[0]
        idx = np.argmax(probs)
        label = le.inverse_transform([idx])[0]
        return jsonify({"label": label, "confidence": float(probs[idx])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """New endpoint - returns emotion classification + LLM response"""
    try:
        if not request.json or "text" not in request.json:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        user_text = request.json["text"]
        
        # Classify emotion
        text_list = [user_text] if isinstance(user_text, str) else user_text
        vec = vectorizer(text_list)
        probs = model.predict(vec, verbose=0)[0]
        idx = np.argmax(probs)
        emotion = le.inverse_transform([idx])[0]
        confidence = float(probs[idx])
        
        # Map emotion to coach type
        coach_type = EMOTION_TO_COACH.get(emotion, 'empathy-focused supporter')
        
        # Get LLM response
        llm_response = get_llm_response(user_text, emotion, coach_type)
        
        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "coach_type": coach_type,
            "response": llm_response
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)