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

# ==================== MINIMAL THEME #E0E0F3 ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'Prompt',sans-serif; }
div[data-testid="stAppViewContainer"], section.main, .stApp { background:#E0E0F3; }
#MainMenu, header, footer { visibility:hidden; }
h1,h2,h3,h4,p,span,li { color:#33335A; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#FFFFFF; border:none; border-radius:24px;
    box-shadow:0 4px 20px rgba(90,90,160,.07);
}
div[data-testid="stMetric"] {
    background:#FFFFFF; border-radius:20px; padding:1.1rem 1.4rem;
    box-shadow:0 4px 20px rgba(90,90,160,.07);
}
div[data-testid="stMetric"] label { color:#8A8AA8 !important; font-weight:500; }
div[data-testid="stTabs"] ul { gap:.4rem; }
div[data-testid="stTabs"] button { background:transparent; color:#6E6E93; border-radius:999px; font-weight:500; }
div[data-testid="stTabs"] button[aria-selected="true"] { background:#6C63FF; color:#fff; }
div.stButton > button {
    background:#6C63FF; color:#fff; border:none; border-radius:14px;
    font-weight:600; padding:.55rem 2.5rem;
}
div.stButton > button:hover { background:#574FE0; }
input, textarea { border-radius:12px !important; border:1px solid #DCDCF0 !important; }
div[data-baseweb="select"] > div { border-radius:12px !important; border:1px solid #DCDCF0 !important; }
div[data-testid="stAlert"] { border-radius:16px; }
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
    st.title("💊 Drug Side Effects Dashboard")
    st.caption("ระบบวิเคราะห์รีวิวผู้ป่วยเพื่อจำแนกระดับผลข้างเคียงยาด้วย Machine Learning")
with h2:
    with st.container(border=True):
        if os.path.exists("my_photo.jpg"):
            st.image("my_photo.jpg")
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
        st.subheader("การกำหนดปัญหา")
        st.write("ผู้ป่วยรีวิวผลข้างเคียงยาไว้ในเว็บ แต่ข้อมูลกระจัดกระจาย → ใช้ ML วิเคราะห์รีวิวเพื่อจำแนกระดับความเสี่ยง (สูง/กลาง/ต่ำ) ช่วยให้แพทย์คัดกรองยาที่ต้องระวังได้เร็วขึ้น")
    with st.container(border=True):
        st.subheader("Dataset : Drug Review (จำลอง)")
        st.write("3,000 รีวิวจำลอง • ฟีเจอร์: ชื่อยา, สภาพโรค, รีวิวข้อความ, usefulCount, rating")
        st.dataframe(make_data(20)[["drugName", "condition", "review", "usefulCount", "rating", "side_effect_level"]],
                     use_container_width=True, hide_index=True)

with t2:
    with st.container(border=True):
        st.subheader("ขั้นตอน Data Preprocessing")
        st.markdown("""
        1. สร้างข้อมูลจำลอง 3,000 แถว (10 ยา × 10 โรค × 15 รีวิว)
        2. ทำความสะอาดข้อความ — ลบ HTML, lowercase, ลบสัญลักษณ์
        3. สร้าง target — แปลง rating (1–10) → side_effect_level (high/moderate/low)
        4. TF-IDF Vectorizer — แปลงข้อความเป็นเวกเตอร์ 300 มิติ
        5. StandardScaler — ปรับสเกล usefulCount + review_len
        6. Split 80/20 Stratified
        """)

with t3:
    mc1, mc2 = st.columns(2)
    with mc1.container(border=True):
        st.markdown("**Logistic Regression**  \nMultinomial logistic + TF-IDF → เหมาะกับข้อความขนาดใหญ่")
    with mc2.container(border=True):
        st.markdown("**Decision Tree**  \nแบ่งกิ่งด้วย Gini impurity ตีความง่ายแต่ overfit ง่ายกับ sparse features")
    with mc1.container(border=True):
        st.markdown("**Random Forest**  \nBagging หลายต้น ลด variance เหมาะกับข้อมูล imbalance")
    with mc2.container(border=True):
        st.markdown("**K-NN**  \nโหวตจาก k เพื่อนบ้านใกล้สุด ต้อง scaling ก่อน (TF-IDF + numeric)")

with t4:
    if comp is not None:
        with st.container(border=True):
            st.dataframe(comp, use_container_width=True, hide_index=True)
    i1, i2 = st.columns(2)
    if os.path.exists("compare.png"): i1.image("compare.png", caption="เปรียบเทียบโมเดล")
    if os.path.exists("cm.png"):      i2.image("cm.png", caption="Confusion Matrix (3×3)")
    if os.path.exists("roc.png"):     st.image("roc.png", caption="ROC Curve (One-vs-Rest)")

with t5:
    with st.container(border=True):
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
            model_name = st.selectbox("เลือกโมเดล", list(models.keys()), index=2)
            drug = st.text_input("ชื่อยา (เช่น Metformin)", "Metformin")
            condition = st.text_input("สภาพโรค (เช่น Diabetes)", "Diabetes")
            review = st.text_area("พิมพ์รีวิวผู้ป่วย (ภาษาอังกฤษ)",
                                  "This drug caused severe nausea and dizziness. I could not continue.",
                                  height=120)
            useful = st.number_input("จำนวน usefulCount", 0, 1000, value=50)

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

                level_map = {"high": "🔴 ผลข้างเคียงรุนแรง",
                             "moderate": "🟡 ผลข้างเคียงปานกลาง",
                             "low": "🟢 ผลข้างเคียงน้อย"}
                st.markdown(f"### {level_map.get(pred, pred)}")
                st.progress(float(proba[idx]))
                st.write(f"**ความมั่นใจ:** {proba[idx]:.1%}")
                with st.expander("ดูความน่าจะเป็นทุกคลาส"):
                    for cls, p in sorted(zip(classes, proba * 100), key=lambda x: -x[1]):
                        st.write(f"- {level_map.get(cls, cls)}: {p:.1f}%")