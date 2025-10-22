import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def load_ml_models():
    log_model = pickle.load(open('models/logistic_model.pkl', 'rb'))
    rf_model = pickle.load(open('models/random_forest.pkl', 'rb'))
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    return log_model, rf_model, vectorizer

def predict_ml(text, model_type='logistic'):
    log_model, rf_model, vectorizer = load_ml_models()
    X_vec = vectorizer.transform([text])
    if model_type == 'logistic':
        pred = log_model.predict(X_vec)
    else:
        pred = rf_model.predict(X_vec)
    return "Real" if pred[0] == 1 else "Fake"

def predict_lstm(text):
    model = load_model('models/lstm_model.h5')
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300, padding='post', truncating='post')
    pred = model.predict(padded)
    return "Real" if pred[0][0] > 0.5 else "Fake"
