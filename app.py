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

st.set_page_config(page_title="ระบบวิเคราะห์ผลข้างเคียงของยา", page_icon="💊", layout="wide")
st.session_state.setdefault("models", None)

# ==================== FORMAL THEME (Sarabun + Navy on #E0E0F3) ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;500;600;700&display=swap');

/* ---------- พื้นฐาน ---------- */
html, body, [class*="css"] { font-family:'Sarabun',sans-serif; }
div[data-testid="stAppViewContainer"], section.main, .stApp { background:#E0E0F3; }
#MainMenu, header, footer { visibility:hidden; }
h1,h2,h3,h4,p,span,li { color:#1F2430; }
h1 { font-weight:700; color:#16204D; }
h2,h3 { font-weight:600; color:#16204D; }

/* ---------- การ์ดเนื้อหา ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#FFFFFF;
    border:1px solid #D5D5E6;
    border-radius:10px;
    box-shadow:0 1px 3px rgba(20,25,60,.06);
}

/* ---------- การ์ดตัวชี้วัด (KPI) ---------- */
div[data-testid="stMetric"] {
    background:#FFFFFF;
    border:1px solid #D5D5E6;
    border-left:4px solid #1F2A63;
    border-radius:8px;
    padding:1rem 1.2rem;
    box-shadow:none;
}
div[data-testid="stMetric"] label { color:#5A6178 !important; font-weight:600; font-size:.85rem; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color:#16204D; font-weight:700; }

/* ---------- Tabs แบบเส้นล่าง (ทางการ) ---------- */
div[data-testid="stTabs"] ul { gap:0; padding:0; background:transparent; border-radius:0; border-bottom:1px solid #C6C6DA; }
div[data-testid="stTabs"] button {
    background:transparent; color:#4A4A68; border-radius:0;
    font-weight:500; padding:.6rem 1.3rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background:transparent; color:#1F2A63; font-weight:700;
    border-bottom:3px solid #1F2A63; box-shadow:none;
}
div[data-testid="stTabs"] button:hover { color:#1F2A63; }

/* ---------- ปุ่ม ---------- */
div.stButton > button {
    background:#1F2A63; color:#FFFFFF; border:none; border-radius:8px;
    font-weight:600; padding:.55rem 2rem; box-shadow:none;
}
div.stButton > button:hover { background:#16204D; }

/* ---------- ช่องกรอก ---------- */
input, textarea { border-radius:8px !important; border:1px solid #C6C6DA !important; }
div[data-baseweb="select"] > div { border-radius:8px !important; border:1px solid #C6C6DA !important; }

/* ---------- แถบความคืบหน้า ---------- */
div[data-testid="stProgress"] > div { background:#E4E4F0; border-radius:6px; }
div[data-testid="stProgress"] > div > div { background:#1F2A63; border-radius:6px; }

/* ---------- ส่วนขยาย ---------- */
div[data-testid="stExpander"] { border-radius:8px; background:#FAFAFE; border:1px solid #D5D5E6; }

/* ---------- ป้ายระดับความเสี่ยง (โทนสุภาพ) ---------- */
.risk-high, .risk-moderate, .risk-low {
    padding:.55rem 1rem; border-radius:8px; font-weight:600; display:inline-block;
}
.risk-high     { background:#FDECEA; color:#B3261E; border:1px solid #F2C4BE; }
.risk-moderate { background:#FFF8E1; color:#8D6E00; border:1px solid #EBDCA6; }
.risk-low      { background:#E8F5E9; color:#2E7D32; border:1px solid #BFE0C3; }

/* ---------- องค์ประกอบหัวเรื่อง ---------- */
.tag-project {
    display:inline-block; background:#1F2A63; color:#FFFFFF;
    font-size:.72rem; letter-spacing:2px; padding:.3rem .8rem; border-radius:4px;
}
.header-line { width:64px; height:3px; background:#1F2A63; margin:.6rem 0 1rem 0; }
.info-table td { padding:.15rem 0; color:#1F2430; }
.info-table td:first-child { color:#5A6178; padding-right:.8rem; }
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

# ==================== ส่วนหัวโครงการ ====================
st.markdown('<span class="tag-project">MACHINE LEARNING PROJECT</span>', unsafe_allow_html=True)
h1, h2 = st.columns([3, 1], gap="large")
with h1:
    st.title("ระบบวิเคราะห์ผลข้างเคียงของยา")
    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
    st.caption("การจำแนกระดับผลข้างเคียงจากข้อความรีวิวผู้ป่วยด้วยเทคนิคการเรียนรู้ของเครื่อง")
with h2:
    with st.container(border=True):
        if os.path.exists("my_photo.jpg"):
            st.image("my_photo.jpg", use_container_width=True)
        st.markdown("""
        <table class="info-table">
          <tr><td>รหัสนักศึกษา</td><td>63xxxxxxxx</td></tr>
          <tr><td>ชื่อ-นามสกุล</td><td>……………</td></tr>
          <tr><td>หมู่เรียน</td><td>……</td></tr>
        </table>
        """, unsafe_allow_html=True)

# ==================== ตัวชี้วัดโครงการ ====================
m1, m2, m3, m4, _ = st.columns([1, 1, 1, 1, 1.6], gap="medium")
m1.metric("ขนาดข้อมูล", "3,000 รายการ")
m2.metric("คุณลักษณะ", "TF-IDF 300 + 2 ตัวเลข")
m3.metric("โมเดลที่ดีที่สุด", best["Model"] if best is not None else "–")
m4.metric("ความถูกต้องสูงสุด", f"{best['Accuracy']:.2%}" if best is not None else "–")

st.markdown("")

# ==================== เนื้อหา 5 ส่วนตามโจทย์ ====================
t1, t2, t3, t4, t5 = st.tabs(["1. ปัญหาและข้อมูล", "2. Preprocessing", "3. โมเดล", "4. การประเมินผล", "5. ทดลองวิเคราะห์"])

with t1:
    with st.container(border=True):
        st.subheader("1.1 การกำหนดปัญหา")
        st.write("ผู้ป่วยที่รับประทานยามักบันทึกประสบการณ์และผลข้างเคียงไว้ในช่องทางออนไลน์ "
                   "แต่ข้อมูลมีปริมาณมากและกระจัดกระจาย งานนี้จึงพัฒนาโมเดลการเรียนรู้ของเครื่องเพื่อจำแนกระดับ"
                   "ผลข้างเคียง (รุนแรง / ปานกลาง / น้อย) จากข้อความรีวิวอัตโนมัติ "
                   "เพื่อสนับสนุนการคัดกรองความปลอดภัยของการใช้ยา")
    with st.container(border=True):
        st.subheader("1.2 ชุดข้อมูล (Drug Review Dataset)")
        st.write("ข้อมูลรีวิวผู้ป่วยจำลอง 3,000 รายการ ประกอบด้วย ชื่อยา สภาพโรค ข้อความรีวิว "
                   "จำนวนผู้พบว่ารีวิวมีประโยชน์ (usefulCount) และคะแนนความพึงพอใจ (rating 1–10)")
        st.dataframe(make_data(10)[["drugName", "condition", "review", "usefulCount", "rating", "side_effect_level"]],
                     use_container_width=True, hide_index=True)

with t2:
    with st.container(border=True):
        st.subheader("2.1 ขั้นตอนการเตรียมข้อมูล")
        st.markdown("""
        1. **การสร้างและคัดเลือกข้อมูล** — สุ่มตัวอย่างรีวิว 3,000 รายการแบบ Stratified
        2. **การทำความสะอาดข้อความ** — ตัดแท็ก HTML แปลงเป็นตัวพิมพ์เล็ก และลบอักขระพิเศษ
        3. **การกำหนดตัวแปรเป้าหมาย** — แปลงคะแนน rating เป็น 3 ระดับ ได้แก่ high / moderate / low
        4. **การแปลงข้อความเป็นคุณลักษณะ** — ใช้ TF-IDF Vectorizer (300 มิติ)
        5. **การปรับมาตราส่วน** — ใช้ StandardScaler กับคุณลักษณะตัวเลข
        6. **การแบ่งข้อมูล** — ชุดฝึกและชุดทดสอบ อัตราส่วน 80:20
        """)

with t3:
    st.subheader("3.1 โมเดลที่ใช้ในการศึกษา")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("**Logistic Regression**  \nแบบจำลองเชิงเส้นที่ใช้ฟังก์ชัน Sigmoid ประมาณความน่าจะเป็นของแต่ละคลาส เหมาะสำหรับข้อมูลข้อความมิติสูง")
        with st.container(border=True):
            st.markdown("**Random Forest**  \nวิธี Ensemble แบบ Bagging ที่สร้าง Decision Tree จำนวนมากและรวมผลด้วยการโหวต เพื่อลดความแปรปรวนของแบบจำลอง")
    with c2:
        with st.container(border=True):
            st.markdown("**Decision Tree**  \nแบ่งข้อมูลเป็นกิ่งตามค่าที่ลดความไม่บริสุทธิ์ (Gini Impurity) ลง ตีความผลลัพธ์ได้ง่าย")
        with st.container(border=True):
            st.markdown("**K-Nearest Neighbors (K-NN)**  \nจำแนกคลาสจากเพื่อนบ้านที่ใกล้ที่สุด k รายด้วยระยะทางแบบยุคลิด จำเป็นต้องปรับมาตราส่วนข้อมูลล่วงหน้า")

with t4:
    if comp is not None:
        with st.container(border=True):
            st.subheader("4.1 ตารางเปรียบเทียบประสิทธิภาพโมเดล")
            st.dataframe(comp, use_container_width=True, hide_index=True)
    i1, i2 = st.columns(2, gap="medium")
    if os.path.exists("compare.png"):
        with i1:
            with st.container(border=True):
                st.image("compare.png", caption="ภาพที่ 1: การเปรียบเทียบประสิทธิภาพของโมเดล", use_container_width=True)
    if os.path.exists("cm.png"):
        with i2:
            with st.container(border=True):
                st.image("cm.png", caption="ภาพที่ 2: Confusion Matrix ของโมเดลที่ดีที่สุด", use_container_width=True)
    if os.path.exists("roc.png"):
        with st.container(border=True):
            st.image("roc.png", caption="ภาพที่ 3: เส้นโค้ง ROC (One-vs-Rest)", use_container_width=True)

with t5:
    with st.container(border=True):
        st.subheader("5.1 ทดลองวิเคราะห์รีวิว")
        if st.session_state.models is None:
            st.info("โมเดลยังไม่ถูกฝึก — กดปุ่มด้านล่างเพื่อเริ่มต้น (ใช้เวลาประมาณ 10–20 วินาที)")
            if st.button("เริ่มต้นฝึกโมเดล", use_container_width=True):
                try:
                    with st.spinner("กำลังสร้างข้อมูลและฝึกโมเดลทั้ง 4 รูปแบบ..."):
                        st.session_state.models = build_models()
                    st.rerun()
                except Exception as e:
                    st.error(f"ไม่สามารถฝึกโมเดลได้: {e}")
        else:
            models = st.session_state.models
            model_name = st.selectbox("เลือกโมเดล", list(models.keys()), index=2)

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                drug = st.text_input("ชื่อยา", "Metformin")
                condition = st.text_input("สภาพโรค", "Diabetes")
                useful = st.number_input("จำนวน usefulCount", 0, 1000, value=50)
            with c2:
                review = st.text_area("ข้อความรีวิว (ภาษาอังกฤษ)",
                                      "This drug caused severe nausea and dizziness. I could not continue.",
                                      height=140)

            if st.button("วิเคราะห์ผล", use_container_width=True):
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
                    "high": ("ระดับรุนแรง (High)", "risk-high"),
                    "moderate": ("ระดับปานกลาง (Moderate)", "risk-moderate"),
                    "low": ("ระดับน้อย (Low)", "risk-low"),
                }
                label, css = level_map.get(pred, (pred, "risk-low"))
                st.markdown(f"**ผลการวิเคราะห์:** &nbsp; <span class='{css}'>{label}</span>", unsafe_allow_html=True)
                st.write(f"ความเชื่อมั่นของโมเดล: **{proba[idx]:.1%}**")
                st.progress(float(proba[idx]))

                with st.expander("ดูความน่าจะเป็นรายคลาส"):
                    for cls, p in sorted(zip(classes, proba * 100), key=lambda x: -x[1]):
                        l, _ = level_map.get(cls, (cls, ""))
                        st.write(f"- {l}: {p:.1f}%")

# ==================== ส่วนท้าย ====================
st.markdown("---")
st.caption("จัดทำเพื่อประกอบการเรียนวิชา Machine Learning • ปีการศึกษา 2568 • พัฒนาด้วย Python, scikit-learn และ Streamlit")