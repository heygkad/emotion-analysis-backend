import tensorflow as tf
import numpy as np
import joblib
import os

# configure tensorflow for mac gpu acceleration
os.environ['TF_METAL_DEVICE_PLACEMENT'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# load the model
print("Loading emotion classification model...")
model = tf.keras.models.load_model("emotion_model.keras")
print("✓ Model loaded successfully!")

# load the vectorizer
print("Loading text vectorizer...")
vectorizer_model = tf.keras.models.load_model("vectorizer.keras")
vectorizer = vectorizer_model.layers[0]  # extract textvectorization layer
print("✓ Vectorizer loaded successfully!")

# load the label encoder
print("Loading label encoder...")
le = joblib.load("label_encoder.pkl")
print("✓ Label encoder loaded successfully!")

# get class names from label encoder
class_names = le.classes_
print(f"\nAvailable emotion classes: {list(class_names)}")
print("=" * 60)

def predict_emotion(text):
    # ensure text is a list
    if isinstance(text, str):
        text = [text]
    
    # vectorize the text
    vectorized = vectorizer(text)
    
    # make prediction
    predictions = model.predict(vectorized, verbose=0)
    
    # get predicted class indices using argmax
    predicted_indices = np.argmax(predictions, axis=1)
    
    # get predicted class names
    predicted_classes = class_names[predicted_indices]
    
    # get confidence scores
    confidence_scores = np.max(predictions, axis=1)
    
    # prepare results with top classification only
    results = []
    for i, (pred_class, confidence) in enumerate(zip(predicted_classes, confidence_scores)):
        results.append({
            'text': text[i],
            'predicted_emotion': pred_class,
            'confidence': float(confidence)
        })
    
    return results if len(results) > 1 else results[0]

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Emotion Classification Inference")
    print("=" * 60)
    
    # example texts to classify
    test_texts = [
        "I'm so happy today! Everything is going great!",
        "I feel really sad and lonely right now.",
        "This is making me so angry!",
        "I love spending time with my family.",
        "I'm scared about what might happen.",
        "That's disgusting, I can't believe it.",
        "It's just a normal day, nothing special."
    ]
    
    print("\nMaking predictions...\n")
    for text in test_texts:
        result = predict_emotion(text)
        print(f"Text: {result['text']}")
        print(f"Predicted Emotion: {result['predicted_emotion']} (confidence: {result['confidence']:.2%})")
        print()
    
    # interactive mode
    print("=" * 60)
    print("Interactive mode - Enter text to classify (or 'quit' to exit)")
    print("=" * 60)
    
    while True:
        user_input = input("\nEnter text: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        result = predict_emotion(user_input)
        print(f"\nPredicted Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")

