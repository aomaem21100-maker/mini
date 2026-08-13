import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CKD Prediction", page_icon="🩺", layout="wide")

model = joblib.load("best_model.pkl")
ref = pd.read_csv("kidney_disease.csv")
ref.columns = ref.columns.str.strip()
ref = ref.drop(columns=["id"], errors="ignore")
for c in ref.select_dtypes(include="object").columns:
    ref[c] = ref[c].str.strip()
for c in ["pcv", "wc", "rc"]:
    ref[c] = pd.to_numeric(ref[c], errors="coerce")
X_ref = ref.drop(columns=["classification"])

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
    st.dataframe(ref.head(10), use_container_width=True)

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
    - **Decision Tree:** แบ่งข้อมูลเป็นกิ่งด้วยฟีเจอร์ที่ลดความไม่บริสุทธิ์ (Gini/Entropy) มากที่สุด ตีความง่ายแต่ Overfit ง่าย
    - **Random Forest:** สร้าง Decision Tree หลายต้นแบบ Bagging แล้วโหวตรวม ลด Overfitting
    - **K-NN:** จำแนกคลาสจากเพื่อนบ้าน k ตัวที่ใกล้ที่สุดด้วยระยะทางยุคลิด จึงต้อง Scaling ก่อนเสมอ
    """)

with t4:
    st.subheader("การประเมินและเปรียบเทียบโมเดล")
    st.dataframe(pd.read_csv("model_comparison.csv"), use_container_width=True)
    st.image("compare.png", caption="กราฟแท่งเปรียบเทียบโมเดล")
    st.image("roc.png", caption="กราฟ ROC Curve")
    st.image("cm.png", caption="Confusion Matrix ของโมเดลที่ดีที่สุด")

with t5:
    st.subheader("ทดลองทำนาย")
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
        pred = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0][1] * 100
        if pred == 1:
            st.error(f"ผลทำนาย: **เสี่ยงโรคไตเรื้อรัง (ckd)** ความมั่นใจ {proba:.1f}%")
        else:
            st.success(f"ผลทำนาย: **ไม่เสี่ยง (notckd)** ความมั่นใจ {100 - proba:.1f}%")