import os
import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Drug Side Effects Dashboard", page_icon="💊", layout="wide")
st.session_state.setdefault("models", None)

# ==================== PREMIUM THEME #E0E0F3 (Enhanced) ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"] { font-family:'Prompt',sans-serif; }
div[data-testid="stAppViewContainer"], section.main, .stApp { 
    background: linear-gradient(135deg, #E0E0F3 0%, #D5D5EB 100%);
}
#MainMenu, header, footer { visibility:hidden; }
h1,h2,h3,h4,p,span,li { color:#33335A; }

/* Typography enhancements */
h1 { font-weight:700; letter-spacing:-0.5px; }
h2 { font-weight:600; letter-spacing:-0.3px; }
h3 { font-weight:600; }

/* Card base with glass morphism */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(90, 90, 160, 0.08);
    transition: all 0.3s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(90, 90, 160, 0.12);
}

/* Metric cards premium */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8F8FD 100%);
    border-radius: 20px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 20px rgba(90, 90, 160, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.1);
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(108, 99, 255, 0.15);
    border-color: rgba(108, 99, 255, 0.3);
}
div[data-testid="stMetric"] label { 
    color:#6E6E93 !important; 
    font-weight:600; 
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-weight: 700;
    color: #33335A;
    font-size: 1.8rem;
}

/* Tabs pill style with gradient */
div[data-testid="stTabs"] ul { gap:.5rem; padding: 0.3rem; background: rgba(255,255,255,0.5); border-radius: 999px; }
div[data-testid="stTabs"] button { 
    background:transparent; 
    color:#6E6E93; 
    border-radius:999px; 
    font-weight:500;
    padding: 0.5rem 1.2rem;
    transition: all 0.3s ease;
}
div[data-testid="stTabs"] button[aria-selected="true"] { 
    background: linear-gradient(135deg, #6C63FF 0%, #574FE0 100%);
    color:#fff;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
}
div[data-testid="stTabs"] button:hover { 
    color:#6C63FF;
    transform: translateY(-1px);
}

/* Primary button with gradient */
div.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #574FE0 100%);
    color:#fff; 
    border:none; 
    border-radius:14px;
    font-weight:600; 
    padding:.6rem 2.5rem;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
    transition: all 0.3s ease;
}
div.stButton > button:hover { 
    background: linear-gradient(135deg, #574FE0 0%, #4A3FD0 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4);
}
div.stButton > button:active {
    transform: translateY(0);
}

/* Input fields */
input, textarea { 
    border-radius:12px !important; 
    border:2px solid rgba(108, 99, 255, 0.15) !important;
    transition: all 0.3s ease !important;
}
input:focus, textarea:focus {
    border-color: rgba(108, 99, 255, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.1) !important;
}
div[data-baseweb="select"] > div { 
    border-radius:12px !important; 
    border:2px solid rgba(108, 99, 255, 0.15) !important;
}

/* Alerts */
div[data-testid="stAlert"] { 
    border-radius:16px;
    border: 1px solid rgba(108, 99, 255, 0.2);
}

/* Progress bar custom */
div[data-testid="stProgress"] > div {
    background: rgba(108, 99, 255, 0.1);
    border-radius: 10px;
}
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6C63FF 0%, #574FE0 100%);
    border-radius: 10px;
}

/* Expander */
div[data-testid="stExpander"] {
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(108, 99, 255, 0.1);
}
div[data-testid="stExpander"] > details > summary {
    padding: 0.8rem 1.2rem;
    font-weight: 500;
}

/* Spinner animation */
@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Fade in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    animation: fadeIn 0.5s ease-out;
}

/* Risk level indicators */
.risk-high {
    background: linear-gradient(135deg, #FF6B6B 0%, #EE5A52 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 12px;
    font-weight: 600;
    display: inline-block;
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
}
.risk-moderate {
    background: linear-gradient(135deg, #FFD93D 0%, #F6C90E 100%);
    color: #333;
    padding: 0.5rem 1rem;
    border-radius: 12px;
    font-weight: 600;
    display: inline-block;
    box-shadow: 0 4px 15px rgba(255, 217, 61, 0.3);
}
.risk-low {
    background: linear-gradient(135deg, #6BCB77 0%, #4CAF50 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 12px;
    font-weight: 600;
    display: inline-block;
    box-shadow: 0 4px 15px rgba(107, 203, 119, 0.3);
}

/* Icon indicators */
.icon-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-right: 0.5rem;
}

/* Model cards */
.model-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8F8FD 100%);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #6C63FF;
    box-shadow: 0 2px 10px rgba(90, 90, 160, 0.05);
}

/* Header accent line */
.header-accent {
    width: 60px;
    height: 4px;
    background: linear-gradient(90deg, #6C63FF 0%, #574FE0 100%);
    border-radius: 2px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ==================== Helpers ====================
class TextExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X["clean_review"].values

class NumericExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X[["usefulCount", "review_len"]].values

def clean_text(text):
    text = re.sub(r"<.*?>", "", str(text))
    text = text.lower()
    return re.sub(r"[^a-z\s]", "", text).strip()

def make_data(n=3000, seed=42):
    np.random.seed(seed)
    drugs = ["Metformin", "Lisinopril", "Atorvastatin", "Omeprazole", "Sertraline",
             "Amlodipine", "Losartan", "Gabapentin", "Tramadol", "Prednisone"]
    conditions = ["Diabetes", "Hypertension", "High Cholesterol", "GERD", "Depression",
                  "Anxiety", "Pain", "Arthritis", "Asthma", "Infection"]
    reviews = [
        "This drug caused severe nausea and dizziness. I could not continue taking it anymore.",
        "Worked well but had mild headache for the first week. Overall acceptable results.",
        "Excellent results with no side effects at all. Highly recommended for patients.",
        "Terrible experience. Vomiting and fatigue every day. Stopped after three days.",
        "Good medication, only minor stomach upset occasionally. Will continue using it.",
        "Caused extreme drowsiness and dry mouth. Not suitable for daily activities.",
        "Helped reduce my symptoms significantly. Very happy with this treatment plan.",
        "Severe allergic reaction. Had to visit emergency room immediately after taking.",
        "Moderate effectiveness with some mild side effects like mild headache occasionally.",
        "Amazing medication. Cleared my condition completely within two weeks of use.",
        "Experienced severe insomnia and anxiety after starting this medication last month.",
        "Mild improvement in symptoms but the cost is too high for long term use.",
        "Caused weight gain and mood swings. Discontinued after consulting with doctor.",
        "Very effective for pain relief but caused constipation as a side effect.",
        "Best medication I have ever taken. No side effects and works quickly.",
    ]
    df = pd.DataFrame({
        "drugName": np.random.choice(drugs, n),
        "condition": np.random.choice(conditions, n),
        "review": np.random.choice(reviews, n),
        "usefulCount": np.random.randint(0, 500, n),
        "rating": np.random.randint(1, 11, n),
    })
    df["side_effect_level"] = df["rating"].apply(
        lambda r: "high" if r <= 3 else ("moderate" if r <= 7 else "low"))
    df["clean_review"] = df["review"].apply(clean_text)
    df["review_len"] = df["clean_review"].str.split().str.len()
    return df

@st.cache_resource
def build_models():
    df = make_data(3000)
    X = df[["clean_review", "usefulCount", "review_len"]]
    y = df["side_effect_level"]

    preprocessor = FeatureUnion([
        ("tfidf", Pipeline([("extract", TextExtractor()),
                            ("tfidf", TfidfVectorizer(max_features=300, stop_words="english"))])),
        ("numeric", Pipeline([("extract", NumericExtractor()),
                              ("scale", StandardScaler())])),
    ])
    models = {
        "Logistic Regression": Pipeline([("pre", preprocessor), ("m", LogisticRegression(max_iter=1000))]),
        "Decision Tree":       Pipeline([("pre", preprocessor), ("m", DecisionTreeClassifier(max_depth=8, random_state=42))]),
        "Random Forest":       Pipeline([("pre", preprocessor), ("m", RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42))]),
        "K-NN":                Pipeline([("pre", preprocessor), ("m", KNeighborsClassifier(n_neighbors=5))]),
    }
    for p in models.values():
        p.fit(X, y)
    return models

comp = pd.read_csv("model_comparison.csv") if os.path.exists("model_comparison.csv") else None
best = comp.sort_values("Accuracy", ascending=False).iloc[0] if comp is not None else None

# ==================== HEADER ====================
h1, h2 = st.columns([3, 1], gap="large")
with h1:
    st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
    st.title("💊 Drug Side Effects Dashboard")
    st.caption("ระบบวิเคราะห์รีวิวผู้ป่วยเพื่อจำแนกระดับผลข้างเคียงยาด้วย Machine Learning")
with h2:
    with st.container(border=True):
        if os.path.exists("my_photo.jpg"):
            st.image("my_photo.jpg", use_column_width=True)
        st.markdown("**รหัส:** 63xxxxxxxx  \n**ชื่อ-นามสกุล:** ……………  \n**หมู่เรียน:** ……")

# ==================== METRICS (ชิดซ้าย) ====================
m1, m2, m3, m4, _ = st.columns([1, 1, 1, 1, 1.6], gap="medium")
m1.metric("ข้อมูลรีวิว", "3,000 รายการ")
m2.metric("ฟีเจอร์", "TF-IDF 300 + 2 numeric")
m3.metric("โมเดลที่ดีที่สุด", best["Model"] if best is not None else "–")
m4.metric("Accuracy สูงสุด", f"{best['Accuracy']:.2%}" if best is not None else "–")

st.markdown("")

# ==================== TABS ====================
t1, t2, t3, t4, t5 = st.tabs(["📌 ปัญหา", "🧹 Preprocessing", "🤖 โมเดล", "📊 ประเมินผล", "🔮 วิเคราะห์รีวิว"])

with t1:
    with st.container(border=True):
        st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
        st.subheader("การกำหนดปัญหา")
        st.write("ผู้ป่วยรีวิวผลข้างเคียงยาไว้ในเว็บ แต่ข้อมูลกระจัดกระจาย → ใช้ ML วิเคราะห์รีวิวเพื่อจำแนกระดับความเสี่ยง (สูง/กลาง/ต่ำ) ช่วยให้แพทย์คัดกรองยาที่ต้องระวังได้เร็วขึ้น")
    with st.container(border=True):
        st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
        st.subheader("Dataset : Drug Review (จำลอง)")
        st.write("3,000 รีวิวจำลอง • ฟีเจอร์: ชื่อยา, สภาพโรค, รีวิวข้อความ, usefulCount, rating")
        st.dataframe(make_data(20)[["drugName", "condition", "review", "usefulCount", "rating", "side_effect_level"]],
                     use_container_width=True, hide_index=True)

with t2:
    with st.container(border=True):
        st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
        st.subheader("ขั้นตอน Data Preprocessing")
        st.markdown("""
        **1. สร้างข้อมูลจำลอง** — 3,000 แถว (10 ยา × 10 โรค × 15 รีวิว)
        
        **2. ทำความสะอาดข้อความ** — ลบ HTML tags, แปลงเป็น lowercase, ลบสัญลักษณ์พิเศษ
        
        **3. สร้างตัวแปรเป้าหมาย** — แปลง rating (1–10) → side_effect_level (high/moderate/low)
        
        **4. TF-IDF Vectorizer** — แปลงข้อความเป็นเวกเตอร์ 300 มิติ
        
        **5. StandardScaler** — ปรับสเกลฟีเจอร์ตัวเลข (usefulCount + review_len)
        
        **6. Split ข้อมูล** — Train/Test = 80/20 แบบ Stratified
        """)

with t3:
    st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
    st.subheader("โมเดล Machine Learning ที่ใช้")
    
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        with st.container(border=True):
            st.markdown("#### 🎯 Logistic Regression")
            st.markdown("Multinomial logistic regression + TF-IDF features  \nเหมาะกับข้อความขนาดใหญ่ คำนวณเร็ว")
    with col2:
        with st.container(border=True):
            st.markdown("#### 🌳 Decision Tree")
            st.markdown("แบ่งกิ่งด้วย Gini impurity  \nตีความง่าย แต่ overfit ได้ง่ายกับ sparse features")
    
    with col1:
        with st.container(border=True):
            st.markdown("#### 🌲 Random Forest")
            st.markdown("Bagging หลายต้นแล้วโหวต  \nลด variance เหมาะกับข้อมูล imbalance")
    with col2:
        with st.container(border=True):
            st.markdown("#### 👥 K-NN")
            st.markdown("โหวตจาก k เพื่อนบ้านใกล้สุด  \nต้อง scaling ก่อน (TF-IDF + numeric)")

with t4:
    if comp is not None:
        with st.container(border=True):
            st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
            st.subheader("ตารางเปรียบเทียบโมเดล")
            st.dataframe(comp, use_container_width=True, hide_index=True)
    
    i1, i2 = st.columns(2, gap="medium")
    if os.path.exists("compare.png"):
        with i1:
            with st.container(border=True):
                st.image("compare.png", caption="กราฟเปรียบเทียบ Performance", use_column_width=True)
    if os.path.exists("cm.png"):
        with i2:
            with st.container(border=True):
                st.image("cm.png", caption="Confusion Matrix (3×3)", use_column_width=True)
    if os.path.exists("roc.png"):
        with st.container(border=True):
            st.image("roc.png", caption="ROC Curve (One-vs-Rest)", use_column_width=True)

with t5:
    with st.container(border=True):
        st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
        st.subheader("ทดลองวิเคราะห์รีวิว")
        
        if st.session_state.models is None:
            st.info("⏳ โมเดลยังไม่ถูกเทรน — กดปุ่มด้านล่างเพื่อเริ่ม (ใช้เวลาประมาณ 10–20 วินาที)")
            if st.button("🚀 เริ่มเทรนโมเดล", use_container_width=True):
                try:
                    with st.spinner("กำลังสร้างข้อมูลและเทรน 4 โมเดล..."):
                        st.session_state.models = build_models()
                    st.rerun()
                except Exception as e:
                    st.error(f"เทรนไม่สำเร็จ: {e}")
        else:
            models = st.session_state.models
            model_name = st.selectbox("เลือกโมเดลที่ต้องการใช้", list(models.keys()), index=2)
            
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                drug = st.text_input("💊 ชื่อยา (เช่น Metformin)", "Metformin")
                condition = st.text_input("🏥 สภาพโรค (เช่น Diabetes)", "Diabetes")
            with col2:
                review = st.text_area("📝 พิมพ์รีวิวผู้ป่วย (ภาษาอังกฤษ)",
                                      "This drug caused severe nausea and dizziness. I could not continue.",
                                      height=120)
                useful = st.number_input("👍 จำนวน usefulCount", 0, 1000, value=50)

            if st.button("🔮 วิเคราะห์ผล", use_container_width=True):
                inp = pd.DataFrame([{
                    "clean_review": clean_text(review),
                    "usefulCount": useful,
                    "review_len": len(clean_text(review).split())
                }])
                m = models[model_name]
                pred = m.predict(inp)[0]
                proba = m.predict_proba(inp)[0]
                classes = list(m.classes_)
                idx = classes.index(pred)

                level_map = {
                    "high": ("🔴 ผลข้างเคียงรุนแรง", "risk-high"),
                    "moderate": ("🟡 ผลข้างเคียงปานกลาง", "risk-moderate"),
                    "low": ("🟢 ผลข้างเคียงน้อย", "risk-low")
                }
                
                label, css_class = level_map.get(pred, (pred, ""))
                st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)
                
                st.write(f"**ความมั่นใจ:** {proba[idx]:.1%}")
                st.progress(float(proba[idx]))
                
                with st.expander("📊 ดูความน่าจะเป็นทุกคลาส"):
                    for cls, p in sorted(zip(classes, proba * 100), key=lambda x: -x[1]):
                        label_cls, _ = level_map.get(cls, (cls, ""))
                        st.write(f"- {label_cls}: **{p:.1f}%**")