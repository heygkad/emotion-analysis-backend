import pandas as pd
import tensorflow as tf
import os
import sys

# configure tensorflow for mac gpu acceleration
os.environ['TF_METAL_DEVICE_PLACEMENT'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# check tensorflow version and metal plugin
print("=" * 60)
print("TensorFlow Configuration Check")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}")

# check python version compatibility
python_version = sys.version_info
if python_version.major == 3 and python_version.minor > 11:
    print(f"\n⚠ WARNING: Python {python_version.major}.{python_version.minor} detected!")
    print("  tensorflow-macos and tensorflow-metal only support Python 3.9-3.11")
    print("  For GPU support, use Python 3.11:")
    print("    conda create -n tf-metal python=3.11")
    print("    conda activate tf-metal")
    print("    pip install tensorflow-macos tensorflow-metal")

# check if tensorflow-metal is installed
try:
    import tensorflow_metal
    print(f"✓ tensorflow-metal is installed")
except ImportError:
    print("⚠ tensorflow-metal is NOT installed!")
    print("  Install it with: pip install tensorflow-metal")
    print("  Note: You need tensorflow-macos AND tensorflow-metal for GPU support")

# list all available devices
print("\nAvailable devices:")
all_devices = tf.config.list_physical_devices()
for device in all_devices:
    print(f"  - {device}")

# check for gpu devices (mps shows up as gpu on mac)
gpu_devices = tf.config.list_physical_devices('GPU')
cpu_devices = tf.config.list_physical_devices('CPU')

print(f"\nGPU devices found: {len(gpu_devices)}")
print(f"CPU devices found: {len(cpu_devices)}")

# check mps availability
if gpu_devices:
    print("\n✓ GPU (MPS) is available and will be used for training")
    for gpu in gpu_devices:
        print(f"  Using: {gpu}")
else:
    print("\n⚠ GPU (MPS) not detected!")
    print("\nTroubleshooting steps:")
    print("1. Ensure you have installed: pip install tensorflow-macos tensorflow-metal")
    print("2. Check that you're using compatible versions (TensorFlow 2.15+ recommended)")
    print("3. Try restarting your Python environment")
    print("4. Verify macOS is up to date")
    print("5. Check if Metal is available: python -c 'import Metal; print(Metal)'")
    print("\nTraining will continue on CPU...")
print("=" * 60)
print()

# load dataset from huggingface
df = pd.read_parquet("hf://datasets/boltuix/emotions-dataset/emotions_dataset.parquet")
print(df.shape)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print(df.shape)

classes_to_keep = {
    'happiness', 'sadness', 'neutral',
    'anger', 'love', 'fear', 'disgust'
}

df = df[df['Label'].isin(classes_to_keep)]

df.reset_index(drop=True, inplace=True)
print(df.shape)

sentences = df['Sentence'].values
labels = df['Label'].values

from sklearn.model_selection import train_test_split

train_sentences, temp_sentences, train_labels, temp_labels = train_test_split(
    sentences, labels, test_size=0.30, stratify=labels, random_state=42
)

val_sentences, test_sentences, val_labels, test_labels = train_test_split(
    temp_sentences, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)

y_train = le.transform(train_labels)
y_val = le.transform(val_labels)
y_test = le.transform(test_labels)

from tensorflow.keras.layers import TextVectorization

MAX_TOKENS = 20000
MAX_LENGTH = 100

vectorizer = TextVectorization(
    max_tokens=20000,
    output_sequence_length=100,
    standardize="lower_and_strip_punctuation"
)

vectorizer.adapt(train_sentences)

padded_train = vectorizer(train_sentences)
padded_val = vectorizer(val_sentences)
padded_test = vectorizer(test_sentences)

import numpy as np

X_train = np.array(padded_train)
X_val = np.array(padded_val)
X_test = np.array(padded_test)

from tensorflow import keras

vocab_size = len(vectorizer.get_vocabulary())

model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_LENGTH),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(7, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[callback],
    verbose=2
)

# save trained model
model.save("emotion_model.keras")

# save vectorizer (build it first by calling on sample data)
import keras
vectorizer_model = keras.models.Sequential([vectorizer])
# build the model by calling it on sample data
_ = vectorizer_model(tf.constant(["sample text"]))
vectorizer_model.save("vectorizer.keras")

# save label encoder
import joblib
joblib.dump(le, "label_encoder.pkl")

print("Saved model, vectorizer, and label encoder successfully!")
