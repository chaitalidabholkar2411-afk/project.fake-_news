import streamlit as st
import pickle

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Fake News Detector", layout="wide")

# -------------------------------
# ADVANCED CSS (PREMIUM UI)
# -------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.6);
}

/* Text area */
textarea {
    background-color: rgba(255,255,255,0.9) !important;
    color: black !important;
    border-radius: 12px !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
}

/* Glass Cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.15);
    border-radius: 15px;
    padding: 15px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Titles */
h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("notebook/model.pkl", "rb"))
vectorizer = pickle.load(open("notebook/vectorizer.pkl", "rb"))

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🧠 AI Dashboard")
st.sidebar.write("Fake News Detection System")

page = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Analyze", "📊 About"])

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_news(text):
    vec = vectorizer.transform([str(text)])

    prob_fake = model.predict_proba(vec)[0][1]
    prob_real = 1 - prob_fake

    label = "FAKE" if prob_fake > 0.5 else "REAL"
    credibility = int(prob_real * 100)

    if prob_fake > 0.7:
        virality = "HIGH 🚀"
        risk = "CRITICAL 🚨"
        action = "🚫 Block Content"
    elif prob_fake > 0.4:
        virality = "MEDIUM ⚠️"
        risk = "MEDIUM ⚠️"
        action = "⚠️ Monitor Content"
    else:
        virality = "LOW ✅"
        risk = "LOW ✅"
        action = "✅ Safe"

    return label, credibility, virality, risk, action


# -------------------------------
# HOME PAGE
# -------------------------------
if page == "🏠 Home":
    st.title("🧠 AI Misinformation Detection System")
    st.markdown("""
    ### 🚀 Features
    - Fake News Detection (ML)
    - Credibility Score
    - Virality Prediction
    - Risk Analysis
    - Action Recommendation
    """)

# -------------------------------
# ANALYZE PAGE
# -------------------------------
elif page == "🔍 Analyze":
    st.title("🔍 Analyze News")

    news = st.text_area("Enter News Text Here", height=200)

    if st.button("Analyze News"):
        if news.strip() == "":
            st.warning("⚠️ Enter some text")
        else:
            label, cred, virality, risk, action = predict_news(news)

            st.subheader("📊 Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Prediction", label)
                st.metric("Credibility", f"{cred}%")

            with col2:
                st.metric("Virality", virality)
                st.metric("Risk", risk)

            with col3:
                st.metric("Action", action)

# -------------------------------
# ABOUT PAGE
# -------------------------------
else:
    st.title("📊 About Project")

    st.markdown("""
    ### 🧠 Description
    AI system to detect misinformation and analyze its impact.

    ### ⚙️ Tech Stack
    - Python
    - Scikit-learn
    - TF-IDF
    - Logistic Regression
    - Streamlit

    ### 🎯 Use Cases
    - Social Media Monitoring
    - Fake News Detection
    - Content Moderation
    """)