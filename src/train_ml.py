import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

from preprocess import clean_text

# ---------------- STEP 1: LOAD DATA ----------------
print("Step 1: Loading dataset...")

df = pd.read_csv("../data/fake_news.csv")
df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

# sample for speed
df = df.sample(3000, random_state=42)

# ---------------- STEP 2: CLEAN TEXT ----------------
print("Step 2: Cleaning text...")
df["text"] = df["text"].apply(clean_text)

# ---------------- STEP 3: SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ---------------- STEP 4: TF-IDF ----------------
print("Step 3: TF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=1500)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ---------------- STEP 5: XGBOOST ----------------
print("Step 4: Training XGBoost...")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train_tfidf, y_train)
xgb_pred = xgb.predict(X_test_tfidf)

print("\n--- XGBoost Evaluation ---")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# ---------------- STEP 6: SVM ----------------
print("Step 5: Training SVM...")

base_svm = LinearSVC()
svm = CalibratedClassifierCV(base_svm, cv=3)
svm.fit(X_train_tfidf, y_train)

svm_pred = svm.predict(X_test_tfidf)

print("\n--- SVM Evaluation ---")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# ---------------- STEP 7: SAVE ----------------
print("Step 6: Saving models...")

pickle.dump(tfidf, open("../models/tfidf.pkl", "wb"))
pickle.dump(xgb, open("../models/xgb_model.pkl", "wb"))
pickle.dump(svm, open("../models/svm_model.pkl", "wb"))

print("\n✅ ML models trained, evaluated, and saved successfully")
