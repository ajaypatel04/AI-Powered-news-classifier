import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocess import clean_text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Powered Social Media Fake News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    tfidf     = pickle.load(open("models/tfidf.pkl", "rb"))
    xgb       = pickle.load(open("models/xgb_model.pkl", "rb"))
    svm       = pickle.load(open("models/svm_model.pkl", "rb"))
    tokenizer = pickle.load(open("models/tokenizer.pkl", "rb"))
    lstm      = load_model("models/lstm_model.h5")
    return tfidf, xgb, svm, tokenizer, lstm

tfidf, xgb, svm, tokenizer, lstm = load_models()

# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #0a0e1a;
    color: white;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* HEADER */
.app-header {
    background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
    padding: 1.5rem 3rem;
}
.header-title { font-size: 2.5rem; font-weight: 800; }

/* TEXTAREA */
[data-testid="stTextArea"] textarea {
    background: rgba(30,41,59,0.85) !important;
    border: 1.5px solid #1e2a40 !important;
    border-radius: 16px !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    padding: 18px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label { display: none !important; }

/* BUTTON */
[data-testid="stButton"] button {
    width: 100% !important;
    background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
    color: white !important;
    border: none !important;
    padding: 1rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.5) !important;
}

/* VERDICT BOX */
.verdict-box {
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 1.5rem 0 1rem;
}
.verdict-box.fake {
    background: linear-gradient(135deg, #dc2626, #991b1b);
    box-shadow: 0 0 60px rgba(220,38,38,0.22);
}
.verdict-box.real {
    background: linear-gradient(135deg, #059669, #047857);
    box-shadow: 0 0 60px rgba(5,150,105,0.22);
}
.verdict-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.verdict-prob {
    font-size: 1.15rem;
    font-weight: 500;
    opacity: 0.88;
    margin-top: 6px;
}
.verdict-prob span {
    font-weight: 800;
    font-size: 1.4rem;
}

/* SECTION LABEL */
.section-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #6366f1;
    margin: 1.2rem 0 0.6rem;
}

/* MODEL CARD — rendered inside st.columns, one per column */
.mcard {
    background: #111827;
    border: 1.5px solid #1e2a40;
    border-radius: 16px;
    padding: 2rem 1.5rem 1.8rem;
    text-align: center;
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
    cursor: default;
    height: 100%;
}
.mcard:hover { transform: translateY(-4px); }

/* Unique glow per model */
.mcard.xgb:hover {
    border-color: #22c55e;
    box-shadow: 0 0 0 1px #22c55e, 0 0 24px 5px rgba(34,197,94,0.28);
}
.mcard.svm:hover {
    border-color: #38bdf8;
    box-shadow: 0 0 0 1px #38bdf8, 0 0 24px 5px rgba(56,189,248,0.28);
}
.mcard.lstm:hover {
    border-color: #a78bfa;
    box-shadow: 0 0 0 1px #a78bfa, 0 0 24px 5px rgba(167,139,250,0.28);
}

.mcard-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 14px;
}
.mcard-pct {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1;
}
.mcard-bar-bg {
    width: 100%;
    height: 5px;
    background: #1e2a40;
    border-radius: 100px;
    margin-top: 16px;
    overflow: hidden;
}
.mcard-bar-fill {
    height: 100%;
    border-radius: 100px;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div class="app-header">
    <div class="header-title">📰 AI Powered Social Media Fake News Classifier</div>
    <div style="font-size:15px;opacity:0.85;margin-top:4px">
        Leveraging advanced machine learning &amp; deep learning to combat misinformation
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ── INPUT ──
_, col_in, _ = st.columns([0.3, 9.4, 0.3])
with col_in:
    st.markdown("### ✍️ Enter News Article to Analyze")
    text = st.text_area(
        "Article input",
        placeholder="Paste or type the news article content here...",
        label_visibility="collapsed",
        height=220,
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 4, 2])
with btn_col:
    analyze_clicked = st.button("🔍 Analyze Article", use_container_width=True)

# ── PREDICTION ──
if analyze_clicked:
    if not text.strip():
        st.warning("⚠️ Please enter some news text to analyze.")
    else:
        with st.spinner("🤖 AI models are analyzing..."):
            cleaned  = clean_text(text)
            vec      = tfidf.transform([cleaned])

            xgb_prob  = float(xgb.predict_proba(vec)[0][1])  * 100
            svm_prob  = float(svm.predict_proba(vec)[0][1])  * 100

            seq      = tokenizer.texts_to_sequences([cleaned])
            pad_seq  = pad_sequences(seq, maxlen=300)
            lstm_prob = float(lstm.predict(pad_seq, verbose=0)[0][0]) * 100

            # Research-backed weights: XGBoost(best) > LSTM(2nd) > SVM(weakest)
            # Based on avg accuracy: XGBoost ~94%, LSTM ~90%, SVM ~85%
            final_prob = round(min(0.4*xgb_prob + 0.4*lstm_prob + 0.2*svm_prob, 100), 2)
            authentic_prob = round(100 - final_prob, 2)

        _, res_col, _ = st.columns([0.3, 9.4, 0.3])
        with res_col:

            # ── Verdict with probability ──
            # needs 60%+ to call fake — prevents borderline misclassification
            if final_prob >= 60:
                v_class    = "fake"
                v_icon     = "🚨"
                v_title    = "Fake News"
                prob_label = "Fake Probability"
                prob_val   = final_prob
            else:
                v_class    = "real"
                v_icon     = "✅"
                v_title    = "Real News"
                prob_label = "Authentic Probability"
                prob_val   = authentic_prob

            st.markdown(f"""
            <div class="verdict-box {v_class}">
                <div class="verdict-title">{v_icon} {v_title}</div>
                <div class="verdict-prob">{prob_label}: <span>{prob_val:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Model cards — one per st.column so HTML renders correctly ──
            st.markdown('<div class="section-label">Individual Model Scores</div>',
                        unsafe_allow_html=True)

            def bar_color(p):
                return "#ef4444" if p >= 60 else ("#eab308" if p >= 40 else "#22c55e")

            def mcard(css, icon, name, pct):
                color = bar_color(pct)
                return f"""
                <div class="mcard {css}">
                    <div class="mcard-label">{icon} {name}</div>
                    <div class="mcard-pct" style="color:{color}">{pct:.1f}%</div>
                    <div class="mcard-bar-bg">
                        <div class="mcard-bar-fill"
                             style="width:{pct:.1f}%;background:{color}"></div>
                    </div>
                </div>"""

            # ✅ Each card in its OWN st.column — this is what fixes raw HTML rendering
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(mcard("xgb",  "🌲", "XGBoost", xgb_prob),  unsafe_allow_html=True)
            with c2:
                st.markdown(mcard("svm",  "🧠", "SVM",     svm_prob),  unsafe_allow_html=True)
            with c3:
                st.markdown(mcard("lstm", "🔁", "LSTM",    lstm_prob), unsafe_allow_html=True)

        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)