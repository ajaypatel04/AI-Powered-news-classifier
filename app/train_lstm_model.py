import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')
fake['label'] = 0
true['label'] = 1
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data['text'].values
y = data['label'].values

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(5000, 64, input_length=300),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[es], batch_size=64)

os.makedirs('models', exist_ok=True)
model.save('models/lstm_model.h5')
print("LSTM model saved successfully.")
