import os
import streamlit as st
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="CKD Prediction", page_icon="🩺", layout="wide")

# ===== เทรนโมเดลสดบน Cloud แทนการโหลด .pkl (กันเวอร์ชันเพี้ยน) =====
@st.cache_resource
def build_models():
    df = pd.read_csv("kidney_disease.csv")
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["id"], errors="ignore")
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    for c in ["pcv", "wc", "rc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X = df.drop(columns=["classification"])
    y = (df["classification"] == "ckd").astype(int)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oe", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                unknown_value=-1))]), cat_cols),
    ])

    models = {
        "Logistic Regression": Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000))]),
        "Decision Tree":       Pipeline([("pre", pre), ("m", DecisionTreeClassifier(random_state=42))]),
        "Random Forest":       Pipeline([("pre", pre), ("m", RandomForestClassifier(random_state=42))]),
        "K-NN":                Pipeline([("pre", pre), ("m", KNeighborsClassifier())]),
    }
    for p in models.values():
        p.fit(X, y)                      # เทรนด้วยข้อมูลทั้งหมดเพื่อใช้ทำนายจริง
    return X, models

X_ref, models = build_models()

# ===== ส่วนบังคับ: หัวข้อ + รูป + รหัส + ชื่อ + หมู่เรียน =====
st.title("🩺 เว็บไซต์ทำนายโรคไตเรื้อรังด้วย Machine Learning")
L, R = st.columns([3, 1])
with R:
    if os.path.exists("my_photo.jpg"):
        st.image("my_photo.jpg", caption="รูปผู้พัฒนา")
    st.markdown("**รหัส:** 63xxxxxxxx  \n**ชื่อ-นามสกุล:** ……………  \n**หมู่เรียน:** ……")
with L:
    st.markdown("ระบบคัดกรองโรคไตเรื้อรัง (CKD) จากค่าแล็บ 24 รายการ ด้วย Machine Learning")

t1, t2, t3, t4, t5 = st.tabs(["📌 ปัญหา&Dataset", "🧹 Preprocessing", "🤖 โมเดล", "📊 ประเมินผล", "🔮 ทำนาย"])

with t1:
    st.subheader("การกำหนดปัญหา")
    st.write("โรคไตเรื้อรังระยะแรกมักไม่มีอาการชัดเจน จึงใช้ ML คัดกรองผู้ป่วยเสี่ยงสูงจากค่าแล็บ เพื่อช่วยแพทย์วินิจฉัยได้เร็วขึ้น")
    st.subheader("Dataset: Chronic Kidney Disease (UCI)")
    st.write("400 แถว | 24 คุณลักษณะ | ตัวแปรเป้าหมาย: classification (ckd / notckd)")
    st.dataframe(X_ref.head(10), use_container_width=True)

with t2:
    st.subheader("ขั้นตอน Data Preprocessing")
    st.markdown("""
    1. แก้ชนิดข้อมูล: แปลงคอลัมน์ pcv, wc, rc จากข้อความ → ตัวเลข
    2. จัดการค่าสูญหาย: ตัวเลขเติม Median / หมวดหมู่เติม Mode
    3. แปลงข้อมูลหมวดหมู่เป็นตัวเลขด้วย Ordinal Encoding
    4. ปรับสเกลข้อมูลตัวเลขด้วย StandardScaler (จำเป็นสำหรับ K-NN)
    5. แบ่งข้อมูล Train/Test = 80/20 แบบ Stratified
    """)

with t3:
    st.subheader("โมเดล Machine Learning ที่ใช้")
    st.markdown("""
    - **Logistic Regression:** ใช้ฟังก์ชัน Sigmoid แปลงค่าเป็นความน่าจะเป็น 0–1 แล้วตัดที่ 0.5 เพื่อจำแนกคลาส
    - **Decision Tree:** แบ่งข้อมูลเป็นกิ่งด้วยฟีเจอร์ที่ลดความไม่บริสุทธิ์ (Gini/Entropy) มากที่สุด
    - **Random Forest:** สร้าง Decision Tree หลายต้นแบบ Bagging แล้วโหวตรวม ลด Overfitting
    - **K-NN:** จำแนกจากเพื่อนบ้าน k ตัวที่ใกล้ที่สุดด้วยระยะทางยุคลิด จึงต้อง Scaling ก่อนเสมอ
    """)

with t4:
    st.subheader("การประเมินและเปรียบเทียบโมเดล")
    if os.path.exists("model_comparison.csv"):
        st.dataframe(pd.read_csv("model_comparison.csv"), use_container_width=True)
    if os.path.exists("compare.png"):
        st.image("compare.png", caption="กราฟแท่งเปรียบเทียบโมเดล")
    if os.path.exists("roc.png"):
        st.image("roc.png", caption="กราฟ ROC Curve")
    if os.path.exists("cm.png"):
        st.image("cm.png", caption="Confusion Matrix ของโมเดลที่ดีที่สุด")

with t5:
    st.subheader("ทดลองทำนาย")
    model_name = st.selectbox("เลือกโมเดลที่ต้องการใช้", list(models.keys()), index=2)
    user_input = {}
    cols = st.columns(4)
    for i, c in enumerate(X_ref.columns):
        with cols[i % 4]:
            if X_ref[c].dtype.kind in "fi":
                user_input[c] = st.number_input(c, value=float(X_ref[c].median()), key=c)
            else:
                opts = sorted(X_ref[c].dropna().unique().tolist())
                user_input[c] = st.selectbox(c, opts, key=c)
    if st.button("🔮 ทำนายผล"):
        inp = pd.DataFrame([user_input])[X_ref.columns]
        m = models[model_name]
        pred = m.predict(inp)[0]
        proba = m.predict_proba(inp)[0][1] * 100
        if pred == 1:
            st.error(f"ผลทำนาย: **เสี่ยงโรคไตเรื้อรัง (ckd)** ความมั่นใจ {proba:.1f}%")
        else:
            st.success(f"ผลทำนาย: **ไม่เสี่ยง (notckd)** ความมั่นใจ {100 - proba:.1f}%")