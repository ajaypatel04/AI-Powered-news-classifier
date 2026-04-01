import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_text

# ---------------- STEP 1: LOAD DATA ----------------
print("Step 1: Loading dataset...")

df = pd.read_csv("../data/fake_news.csv")
df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

print("Original dataset size:", df.shape)

# 🔥 Increase data for better LSTM learning
df = df.sample(10000, random_state=42)
print("Using sample size:", df.shape)

# ---------------- STEP 2: CLEAN TEXT ----------------
print("Step 2: Cleaning text...")
df["text"] = df["text"].apply(clean_text)

# ---------------- STEP 3: TRAIN-TEST SPLIT ----------------
print("Step 3: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ---------------- STEP 4: TOKENIZATION ----------------
print("Step 4: Tokenizing text...")

tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=300)
X_test_pad = pad_sequences(X_test_seq, maxlen=300)

# Save tokenizer
pickle.dump(tokenizer, open("../models/tokenizer.pkl", "wb"))

# ---------------- STEP 5: CLASS WEIGHTS ----------------
print("Step 5: Computing class weights...")

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# ---------------- STEP 6: BUILD LSTM MODEL ----------------
print("Step 6: Building LSTM model...")

model = Sequential()
model.add(Embedding(input_dim=8000, output_dim=128, input_length=300))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# ---------------- STEP 7: TRAIN MODEL ----------------
print("Step 7: Training LSTM model...")

model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test_pad, y_test),
    class_weight=class_weights
)

# ---------------- STEP 8: EVALUATE ----------------
print("Step 8: Evaluating model...")

loss, accuracy = model.evaluate(X_test_pad, y_test)
print("LSTM Test Accuracy:", accuracy)

# ---------------- STEP 9: SAVE MODEL ----------------
print("Step 9: Saving model...")

model.save("../models/lstm_model.h5")

print("\n✅ LSTM model trained, evaluated, and saved successfully")
