import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="ระบบตรวจจับธุรกรรมที่น่าสงสัย", page_icon="💳", layout="wide")
st.session_state.setdefault("models", None)
st.session_state.setdefault("scaler", None)

# ==================== Dialog ผู้พัฒนา ====================
@st.dialog("ข้อมูลผู้พัฒนา")
def show_developer_info():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### ผู้พัฒนา")
        st.write("**นาย จตุรภัทร สถาปีตานนท์**")
        st.write("รหัสนักศึกษา: 664245024")
        st.write("หมู่เรียน: 66/43")
        st.write("สาขา: วิทยาการคอมพิวเตอร์")
    
    with col2:
        st.markdown("### เกี่ยวกับโปรเจกต์")
        st.write("โครงงานนี้เป็นส่วนหนึ่งของวิชา Machine Learning โดยมีวัตถุประสงค์เพื่อศึกษาและประยุกต์ใช้"
                 "เทคนิคการเรียนรู้ของเครื่องในการตรวจจับธุรกรรมบัตรเครดิตที่น่าสงสัย")
        
        st.markdown("### เทคโนโลยีที่ใช้")
        st.write("- Python")
        st.write("- scikit-learn")
        st.write("- Streamlit")
        st.write("- imbalanced-learn")
        
        st.markdown("### ลิงก์ที่เกี่ยวข้อง")
        st.markdown("- [GitHub Profile](https://github.com/aomaem21100-maker?tab=repositories)")
        st.markdown("- [Dataset: Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)")

# ==================== Helpers ====================
def make_data(n=20000, seed=42):
    np.random.seed(seed)
    pca_cols = [f"V{i}" for i in range(1, 29)]
    data = {col: np.random.randn(n) for col in pca_cols}
    data["Time"] = np.random.uniform(0, 172792, n)
    data["Amount"] = np.random.exponential(88, n)
    data["Class"] = np.random.choice([0, 1], n, p=[0.9983, 0.0017])
    return pd.DataFrame(data)

@st.cache_resource
def build_models():
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    df = make_data(20000)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    cols_to_scale = ["Amount", "Time"]
    scaler = StandardScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "K-NN": KNeighborsClassifier(n_neighbors=5),
    }

    trained = {}
    for name, m in models.items():
        if name == "K-NN":
            m.fit(X_train, y_train)
        else:
            m.fit(X_train_res, y_train_res)
        trained[name] = m
    return trained, scaler

comp = pd.read_csv("model_comparison.csv") if os.path.exists("model_comparison.csv") else None
best = comp.sort_values("F1", ascending=False).iloc[0] if comp is not None else None

# ==================== HEADER ====================
st.title("ระบบตรวจจับธุรกรรมที่น่าสงสัย")
st.caption("การจำแนกธุรกรรมปกติและธุรกรรมทุจริตด้วยเทคนิคการเรียนรู้ของเครื่อง")

# ข้อมูลผู้พัฒนา
col1, col2 = st.columns([3, 1])
with col2:
    st.markdown("**ผู้พัฒนา:** นาย จตุรภัทร สถาปีตานนท์")
    st.markdown("**รหัส:** 664245024 | **หมู่เรียน:** 66/43")
    if st.button("ดูข้อมูลเพิ่มเติม"):
        show_developer_info()

st.markdown("---")

# ==================== METRICS ====================
m1, m2, m3, m4 = st.columns(4)
m1.metric("ขนาดข้อมูล", "20,000 รายการ")
m2.metric("คุณลักษณะ", "30 ตัว")
m3.metric("โมเดลที่ดีที่สุด", best["Model"] if best is not None else "–")
m4.metric("F1-Score", f"{best['F1']:.2%}" if best is not None else "–")

st.markdown("---")

# ==================== TABS ====================
t1, t2, t3, t4, t5 = st.tabs([
    "1. ปัญหาและข้อมูล",
    "2. Preprocessing",
    "3. โมเดล",
    "4. การประเมินผล",
    "5. ทดลองตรวจจับ"
])

with t1:
    st.subheader("1.1 การกำหนดปัญหา")
    st.write("ธุรกรรมบัตรเครดิตที่ผิดปกติ (fraud) สร้างความเสียหายทางการเงินมหาศาล "
             "แต่การตรวจสอบด้วยมนุษย์ทำได้ช้าและมีค่าใช้จ่ายสูง งานนี้จึงพัฒนาโมเดลการเรียนรู้ของเครื่องเพื่อตรวจจับ "
             "pattern ของธุรกรรมที่น่าสงสัยแบบอัตโนมัติ")
    
    st.subheader("1.2 ชุดข้อมูล")
    st.write("ข้อมูลธุรกรรมบัตรเครดิต 20,000 รายการ ประกอบด้วย 28 คุณลักษณะจาก PCA, "
             "เวลา (Time), จำนวนเงิน (Amount) และตัวแปรเป้าหมาย Class (0=ปกติ, 1=fraud)")
    st.dataframe(make_data(10), use_container_width=True, hide_index=True)

with t2:
    st.subheader("2.1 ขั้นตอนการเตรียมข้อมูล")
    st.markdown("""
    1. **การสุ่มตัวอย่าง** — ลดขนาดจาก 284,807 เป็น 20,000 รายการ แบบ Stratified
    
    2. **การจัดการ Imbalance** — ใช้ SMOTE เพื่อเพิ่มจำนวน fraud samples ในชุดฝึก
    
    3. **การปรับมาตราส่วน** — ใช้ StandardScaler กับ Amount และ Time
    
    4. **การแบ่งข้อมูล** — Train/Test = 80:20 แบบ Stratified
    
    5. **การเลือกเมตริก** — เน้น Precision, Recall, F1-Score
    """)

with t3:
    st.subheader("3.1 โมเดลที่ใช้ในการศึกษา")
    
    st.markdown("**Logistic Regression**")
    st.caption("แบบจำลองเชิงเส้นสำหรับจำแนกไบนารี ใช้ class_weight='balanced'")
    
    st.markdown("**Decision Tree**")
    st.caption("แบ่งกิ่งตามค่าที่ลด Gini Impurity ตีความง่าย")
    
    st.markdown("**Random Forest**")
    st.caption("Ensemble แบบ Bagging ลด variance พร้อม class_weight")
    
    st.markdown("**K-Nearest Neighbors (K-NN)**")
    st.caption("จำแนกจาก k เพื่อนบ้านที่ใกล้ที่สุด ต้อง scaling ก่อน")

with t4:
    if comp is not None:
        st.subheader("4.1 ตารางเปรียบเทียบประสิทธิภาพ")
        st.dataframe(comp, use_container_width=True, hide_index=True)
    
    if os.path.exists("compare.png"):
        st.image("compare.png", caption="กราฟเปรียบเทียบประสิทธิภาพโมเดล", use_container_width=True)
    
    if os.path.exists("cm.png"):
        st.image("cm.png", caption="Confusion Matrix ของโมเดลที่ดีที่สุด", use_container_width=True)
    
    if os.path.exists("roc.png"):
        st.image("roc.png", caption="เส้นโค้ง ROC", use_container_width=True)
    
    if os.path.exists("pr_curve.png"):
        st.image("pr_curve.png", caption="เส้นโค้ง Precision-Recall", use_container_width=True)

with t5:
    st.subheader("5.1 ทดลองตรวจจับธุรกรรม")
    
    if st.session_state.models is None:
        st.info("โมเดลยังไม่ถูกฝึก — กดปุ่มด้านล่างเพื่อเริ่มต้น")
        if st.button("เริ่มต้นฝึกโมเดล"):
            try:
                with st.spinner("กำลังสร้างข้อมูลและฝึกโมเดล..."):
                    models, scaler = build_models()
                    st.session_state.models = models
                    st.session_state.scaler = scaler
                st.rerun()
            except Exception as e:
                st.error(f"ไม่สามารถฝึกโมเดลได้: {e}")
    else:
        models = st.session_state.models
        scaler = st.session_state.scaler
        model_name = st.selectbox("เลือกโมเดล", list(models.keys()), index=2)

        st.markdown("**กรอกข้อมูลธุรกรรม**")
        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input("Time (วินาที)", 0.0, 200000.0, value=50000.0)
            amount = st.number_input("Amount (USD)", 0.0, 50000.0, value=100.0)
        with c2:
            v1 = st.number_input("V1 (PCA)", -50.0, 50.0, value=0.0)
            v2 = st.number_input("V2 (PCA)", -50.0, 50.0, value=0.0)

        if st.button("ตรวจจับธุรกรรม"):
            inp_dict = {f"V{i}": 0.0 for i in range(1, 29)}
            inp_dict["V1"] = float(v1)
            inp_dict["V2"] = float(v2)

            scaled = scaler.transform(
                pd.DataFrame([[amount, time_val]], columns=["Amount", "Time"])
            )[0]
            inp_dict["Amount"] = float(scaled[0])
            inp_dict["Time"] = float(scaled[1])

            train_columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
            inp = pd.DataFrame([inp_dict])[train_columns]

            m = models[model_name]
            pred = m.predict(inp)[0]
            proba = m.predict_proba(inp)[0][1]

            st.markdown("---")
            if pred == 1:
                st.error(f"**ผลการตรวจจับ:** ธุรกรรมน่าสงสัย (Fraud)")
                st.write(f"ความน่าจะเป็น fraud: {proba:.1%}")
                st.progress(float(proba))
            else:
                st.success(f"**ผลการตรวจจับ:** ธุรกรรมปกติ (Normal)")
                st.write(f"ความน่าจะเป็น fraud: {proba:.1%}")
                st.progress(float(proba))

# ==================== FOOTER ====================
st.markdown("---")
st.caption("จัดทำเพื่อประกอบการเรียนวิชา Machine Learning • พัฒนาด้วย Python, scikit-learn, Streamlit")
st.markdown("[GitHub Repositories](https://github.com/aomaem21100-maker?tab=repositories)")