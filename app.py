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

# ==================== THEME #BFCED6 (Soft Blue-Gray) ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* ===== สีหลัก ===== */
:root {
    --bg-main: #BFCED6;        /* พื้นหลังหลัก */
    --bg-soft: #D4DEE4;        /* พื้นหลังอ่อน */
    --bg-card: #FFFFFF;        /* การ์ดขาว */
    --accent: #2C4A5C;         /* กรมท่า (ปุ่ม/หัวข้อ) */
    --accent-hover: #1F3A4D;   /* กรมท่าเข้ม (hover) */
    --accent-soft: #8BA4B3;    /* ฟ้าเทาอ่อน */
    --text: #1A1F2E;           /* ข้อความหลัก */
    --text-muted: #5A6B7A;     /* ข้อความรอง */
    --border: #A8B8C2;         /* เส้นขอบ */
}

/* ===== พื้นฐาน ===== */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Thai', 'Inter', sans-serif;
}
div[data-testid="stAppViewContainer"], section.main, .stApp {
    background: #BFCED6;
}
#MainMenu, header, footer { visibility: hidden; }

h1, h2, h3, h4 { color: #1A1F2E; font-weight: 600; letter-spacing: -0.3px; }
h1 { font-size: 2rem; font-weight: 700; color: #2C4A5C; }
h2 { font-size: 1.3rem; }
h3 { font-size: 1.1rem; }
p, li { color: #1A1F2E; }
caption, small { color: #5A6B7A !important; }

/* ===== การ์ด ===== */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #D4DEE4;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(44, 74, 92, 0.06);
    padding: 1.3rem;
}

/* ===== Metric Cards ===== */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D4DEE4;
    border-left: 4px solid #2C4A5C;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
div[data-testid="stMetric"] label {
    color: #5A6B7A !important;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #2C4A5C;
    font-weight: 700;
    font-size: 1.5rem;
}

/* ===== Tabs ===== */
div[data-testid="stTabs"] ul {
    gap: 0;
    padding: 0;
    background: transparent;
    border-bottom: 2px solid #A8B8C2;
}
div[data-testid="stTabs"] button {
    background: transparent;
    color: #5A6B7A;
    border-radius: 0;
    font-weight: 500;
    padding: 0.7rem 1.3rem;
    border: none;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2C4A5C;
    border-bottom: 3px solid #2C4A5C;
    font-weight: 600;
    background: transparent;
}
div[data-testid="stTabs"] button:hover {
    color: #2C4A5C;
    background: transparent;
}

/* ===== ปุ่มหลัก ===== */
div.stButton > button {
    background: #2C4A5C;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.8rem;
    box-shadow: 0 1px 2px rgba(44, 74, 92, 0.15);
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: #1F3A4D;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(44, 74, 92, 0.25);
}

/* ===== Input ===== */
input, textarea {
    border-radius: 8px !important;
    border: 1px solid #A8B8C2 !important;
    background: #FFFFFF !important;
    padding: 0.5rem 0.8rem !important;
    color: #1A1F2E !important;
}
input:focus, textarea:focus {
    border-color: #2C4A5C !important;
    box-shadow: 0 0 0 3px rgba(44, 74, 92, 0.1) !important;
}
div[data-baseweb="select"] > div {
    border-radius: 8px !important;
    border: 1px solid #A8B8C2 !important;
    background: #FFFFFF !important;
}

/* ===== Progress Bar ===== */
div[data-testid="stProgress"] > div {
    background: #D4DEE4;
    border-radius: 6px;
}
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #2C4A5C, #3E6478);
    border-radius: 6px;
}

/* ===== Expander ===== */
div[data-testid="stExpander"] {
    border-radius: 10px;
    background: #FFFFFF;
    border: 1px solid #D4DEE4;
}

/* ===== Success / Error / Info boxes ===== */
div[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid #D4DEE4;
}

/* ===== Horizontal Rule ===== */
hr {
    border-color: #A8B8C2 !important;
    margin: 1.5rem 0 !important;
}

/* ===== Dialog / Modal ===== */
div[data-testid="stModal"] {
    background: rgba(44, 74, 92, 0.4) !important;
}
div[data-testid="stModal"] > div {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.8rem;
}

/* ===== Header Accent ===== */
.header-accent {
    width: 56px;
    height: 3px;
    background: #2C4A5C;
    border-radius: 2px;
    margin: 0.4rem 0 1rem 0;
}

/* ===== Project Tag ===== */
.tag-project {
    display: inline-block;
    background: #2C4A5C;
    color: #FFFFFF;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* ===== Dev Card ===== */
.dev-box {
    background: #FFFFFF;
    border: 1px solid #D4DEE4;
    border-radius: 12px;
    padding: 1rem;
}
.dev-label {
    color: #5A6B7A;
    font-size: 0.85rem;
    font-weight: 500;
}
.dev-value {
    color: #1A1F2E;
    font-weight: 600;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

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
st.markdown('<span class="tag-project">MACHINE LEARNING PROJECT</span>', unsafe_allow_html=True)
st.title("ระบบตรวจจับธุรกรรมที่น่าสงสัย")
st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
st.caption("การจำแนกธุรกรรมปกติและธุรกรรมทุจริตด้วยเทคนิคการเรียนรู้ของเครื่อง")

# ข้อมูลผู้พัฒนา
col1, col2 = st.columns([3, 1])
with col2:
    with st.container(border=True):
        st.markdown('<div class="dev-label">ผู้พัฒนา</div>', unsafe_allow_html=True)
        st.markdown('<div class="dev-value">นาย จตุรภัทร สถาปีตานนท์</div>', unsafe_allow_html=True)
        st.markdown('<div class="dev-label" style="margin-top:0.3rem;">รหัส 664245024 • หมู่เรียน 66/43</div>', unsafe_allow_html=True)
        if st.button("ดูข้อมูลเพิ่มเติม", use_container_width=True):
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
    with st.container(border=True):
        st.subheader("1.1 การกำหนดปัญหา")
        st.write("ธุรกรรมบัตรเครดิตที่ผิดปกติ (fraud) สร้างความเสียหายทางการเงินมหาศาล "
                 "แต่การตรวจสอบด้วยมนุษย์ทำได้ช้าและมีค่าใช้จ่ายสูง งานนี้จึงพัฒนาโมเดลการเรียนรู้ของเครื่องเพื่อตรวจจับ "
                 "pattern ของธุรกรรมที่น่าสงสัยแบบอัตโนมัติ")
    
    with st.container(border=True):
        st.subheader("1.2 ชุดข้อมูล")
        st.write("ข้อมูลธุรกรรมบัตรเครดิต 20,000 รายการ ประกอบด้วย 28 คุณลักษณะจาก PCA, "
                 "เวลา (Time), จำนวนเงิน (Amount) และตัวแปรเป้าหมาย Class (0=ปกติ, 1=fraud)")
        st.dataframe(make_data(10), use_container_width=True, hide_index=True)

with t2:
    with st.container(border=True):
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
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Logistic Regression**")
            st.caption("แบบจำลองเชิงเส้นสำหรับจำแนกไบนารี ใช้ class_weight='balanced'")
        with st.container(border=True):
            st.markdown("**Random Forest**")
            st.caption("Ensemble แบบ Bagging ลด variance พร้อม class_weight")
    with c2:
        with st.container(border=True):
            st.markdown("**Decision Tree**")
            st.caption("แบ่งกิ่งตามค่าที่ลด Gini Impurity ตีความง่าย")
        with st.container(border=True):
            st.markdown("**K-Nearest Neighbors (K-NN)**")
            st.caption("จำแนกจาก k เพื่อนบ้านที่ใกล้ที่สุด ต้อง scaling ก่อน")

with t4:
    if comp is not None:
        with st.container(border=True):
            st.subheader("4.1 ตารางเปรียบเทียบประสิทธิภาพ")
            st.dataframe(comp, use_container_width=True, hide_index=True)
    
    if os.path.exists("compare.png"):
        with st.container(border=True):
            st.image("compare.png", caption="กราฟเปรียบเทียบประสิทธิภาพโมเดล", use_container_width=True)
    
    if os.path.exists("cm.png"):
        with st.container(border=True):
            st.image("cm.png", caption="Confusion Matrix ของโมเดลที่ดีที่สุด", use_container_width=True)
    
    if os.path.exists("roc.png"):
        with st.container(border=True):
            st.image("roc.png", caption="เส้นโค้ง ROC", use_container_width=True)
    
    if os.path.exists("pr_curve.png"):
        with st.container(border=True):
            st.image("pr_curve.png", caption="เส้นโค้ง Precision-Recall", use_container_width=True)

with t5:
    with st.container(border=True):
        st.subheader("5.1 ทดลองตรวจจับธุรกรรม")
        
        if st.session_state.models is None:
            st.info("โมเดลยังไม่ถูกฝึก — กดปุ่มด้านล่างเพื่อเริ่มต้น")
            if st.button("เริ่มต้นฝึกโมเดล", use_container_width=True):
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

            if st.button("ตรวจจับธุรกรรม", use_container_width=True):
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
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.caption("จัดทำเพื่อประกอบการเรียนวิชา Machine Learning • พัฒนาด้วย Python, scikit-learn, Streamlit")
with footer_col2:
    st.markdown("<div style='text-align:right;'><a href='https://github.com/aomaem21100-maker?tab=repositories' target='_blank' style='color:#2C4A5C;'>GitHub Repositories →</a></div>", unsafe_allow_html=True)