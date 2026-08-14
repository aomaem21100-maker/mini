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

# ==================== MINIMORE THEME (#E3EBFD) ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', 'Inter', sans-serif; }
div[data-testid="stAppViewContainer"], section.main, .stApp { background: #E3EBFD; }
#MainMenu, header, footer { visibility: hidden; }

/* ✅ ข้อความหลักสีดำ */
h1, h2, h3, h4 { color: #111827; font-weight: 600; letter-spacing: -0.3px; }
h1 { font-size: 2rem; font-weight: 700; }
h2 { font-size: 1.35rem; }
h3 { font-size: 1.1rem; }
p, li { color: #111827; }

/* ✅ บังคับข้อความใน Dialog/Modal เป็นสีดำทั้งหมด */
div[data-testid="stModal"] h1, div[data-testid="stModal"] h2,
div[data-testid="stModal"] h3, div[data-testid="stModal"] h4,
div[data-testid="stModal"] p,  div[data-testid="stModal"] li,
div[data-testid="stModal"] span {
    color: #111827 !important;
}
div[data-testid="stModal"] h3, div[data-testid="stModal"] h4 {
    font-weight: 700 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
    box-shadow: 0 1px 3px rgba(59, 91, 219, 0.04); padding: 1.5rem;
}
div[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
    padding: 1rem 1.2rem; box-shadow: none;
}
div[data-testid="stMetric"] label {
    color: #374151 !important; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #111827; font-weight: 700; font-size: 1.6rem;
}

div[data-testid="stTabs"] ul {
    gap: 0.25rem; padding: 0; background: transparent;
    border-bottom: 1px solid #D1D5DB;
}
div[data-testid="stTabs"] button {
    background: transparent; color: #4B5563; border-radius: 0;
    font-weight: 500; padding: 0.7rem 1.2rem; border: none;
    border-bottom: 2px solid transparent;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #3B5BDB; border-bottom: 2px solid #3B5BDB;
    font-weight: 600; background: transparent; box-shadow: none;
}
div[data-testid="stTabs"] button:hover { color: #3B5BDB; background: transparent; }

div.stButton > button {
    background: #3B5BDB; color: #FFFFFF; border: none; border-radius: 10px;
    font-weight: 500; padding: 0.55rem 1.8rem;
    box-shadow: 0 1px 2px rgba(59, 91, 219, 0.15); transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: #2C4BC4; transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(59, 91, 219, 0.2);
}

input, textarea {
    border-radius: 8px !important; border: 1px solid #D1D5DB !important;
    padding: 0.5rem 0.8rem !important; color: #111827 !important;
}
div[data-baseweb="select"] > div {
    border-radius: 8px !important; border: 1px solid #D1D5DB !important;
}

div[data-testid="stProgress"] > div { background: #E3EBFD; border-radius: 6px; }
div[data-testid="stProgress"] > div > div { background: #3B5BDB; border-radius: 6px; }
div[data-testid="stExpander"] { border-radius: 10px; background: #FFFFFF; border: 1px solid #E5E7EB; }

.risk-fraud, .risk-normal {
    padding: 0.4rem 0.9rem; border-radius: 20px; font-weight: 600;
    font-size: 0.85rem; display: inline-block;
}
.risk-fraud  { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
.risk-normal { background: #D1FAE5; color: #065F46; border: 1px solid #6EE7B7; }

/* ✅ การ์ดผู้พัฒนา (สมดุล) */
.dev-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
    padding: 1rem;
}
.dev-photo-img {
    width: 100%; aspect-ratio: 1; object-fit: cover;
    border-radius: 14px; border: 3px solid #E3EBFD;
}
.photo-placeholder {
    width: 100%; aspect-ratio: 1; background: #E3EBFD; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 3rem; color: #3B5BDB;
}
.dev-name-center { text-align: center; font-weight: 700; color: #111827; margin-top: .6rem; font-size: .95rem; }
.dev-id-center   { text-align: center; color: #111827; font-size: .82rem; margin-top: .15rem; }

.dev-btn {
    background: #E3EBFD; color: #3B5BDB; border: none;
    padding: 0.45rem 0.8rem; border-radius: 8px; font-size: 0.82rem;
    font-weight: 600; display: flex; align-items: center; justify-content: center;
    gap: 0.3rem; text-decoration: none; transition: all 0.2s ease; margin-top: .5rem;
}
.dev-btn:hover { background: #3B5BDB; color: #FFFFFF; }

/* ✅ ป้ายเทคโนโลยีสูงเท่ากัน (สมดุล) */
.tech-badge {
    background: #E3EBFD; color: #3B5BDB; font-weight: 600;
    min-height: 3.6rem; display: flex; align-items: center; justify-content: center;
    text-align: center; border-radius: 10px; padding: .4rem .5rem; font-size: .85rem;
}

/* ✅ แถวข้อมูล: ป้าย + ค่า สีดำ */
.info-row { display: flex; gap: 0.6rem; padding: 0.35rem 0; font-size: 0.92rem; }
.info-row .label { color: #111827; min-width: 130px; font-weight: 600; }
.info-row .value { color: #111827; font-weight: 500; }

.tag-project {
    display: inline-block; background: #FFFFFF; color: #3B5BDB;
    font-size: 0.7rem; letter-spacing: 1.5px; padding: 0.25rem 0.7rem;
    border-radius: 6px; font-weight: 600; border: 1px solid #A5B4FC; margin-bottom: 0.5rem;
}
.header-line { width: 48px; height: 3px; background: #3B5BDB; border-radius: 2px; margin: 0.5rem 0 1rem 0; }

div[data-testid="stModal"] { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# ==================== ✅ ค้นหารูปอัตโนมัติ (แก้รูปไม่ขึ้น) ====================
def find_photo():
    """ค้นหารูปโปรไฟล์ในโฟลเดอร์อัตโนมัติ ไม่ว่าตั้งชื่ออะไร"""
    try:
        files = os.listdir(".")
    except Exception:
        return None
    preferred = ["my_photo.jpg", "my_photo.png", "photo.jpg", "photo.png",
                 "profile.jpg", "profile.png", "me.jpg", "me.png"]
    for name in preferred:
        if name in files:
            return name
    exclude = ("compare", "cm", "roc", "pr_curve", "confusion", "icon", "logo")
    for f in sorted(files):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) \
           and not any(k in f.lower() for k in exclude):
            return f
    return None

PHOTO = find_photo()

# ==================== Dialog ผู้พัฒนา ====================
@st.dialog("👤 ข้อมูลผู้พัฒนา", width="large")
def show_developer_info():
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        if PHOTO:
            st.image(PHOTO, use_container_width=True)
        else:
            st.markdown('<div class="photo-placeholder">👤</div>', unsafe_allow_html=True)
        st.markdown('<div class="dev-name-center">นาย จตุรภัทร สถาปีตานนท์</div>', unsafe_allow_html=True)
        st.markdown('<div class="dev-id-center">🆔 664245024 • 📚 66/43</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### 📇 ข้อมูลส่วนตัว")
        st.markdown("""
        <div class="info-row"><span class="label">🆔 รหัสนักศึกษา</span><span class="value">664245024</span></div>
        <div class="info-row"><span class="label">👤 ชื่อ-นามสกุล</span><span class="value">นาย จตุรภัทร สถาปีตานนท์</span></div>
        <div class="info-row"><span class="label">📚 หมู่เรียน</span><span class="value">66/43</span></div>
        <div class="info-row"><span class="label">🎓 สาขา</span><span class="value">วิทยาการคอมพิวเตอร์</span></div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📝 เกี่ยวกับโปรเจกต์")
    st.write("💡 โครงงานนี้เป็นส่วนหนึ่งของวิชา Machine Learning โดยมีวัตถุประสงค์เพื่อศึกษาและประยุกต์ใช้"
             "เทคนิคการเรียนรู้ของเครื่องในการตรวจจับธุรกรรมบัตรเครดิตที่น่าสงสัย")

    st.markdown("### 🛠️ เทคโนโลยีที่ใช้")
    tech_cols = st.columns(4, gap="small")
    for i, tech in enumerate(["🐍 Python", "🤖 scikit-learn", "🚀 Streamlit", "⚖️ imbalanced-learn"]):
        with tech_cols[i]:
            st.markdown(f'<div class="tech-badge">{tech}</div>', unsafe_allow_html=True)

    st.markdown("### 🔗 ลิงก์ที่เกี่ยวข้อง")
    st.markdown("- 🌐 [GitHub Profile](https://github.com/aomaem21100-maker?tab=repositories)\n"
                "- 📦 [Source Code ของโปรเจกต์นี้](https://github.com/aomaem21100-maker)\n"
                "- 📚 [Dataset: Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)")

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
st.markdown('<span class="tag-project">💳 MACHINE LEARNING PROJECT</span>', unsafe_allow_html=True)

h1, h2 = st.columns([3, 1], gap="large")
with h1:
    st.title("🛡️ ระบบตรวจจับธุรกรรมที่น่าสงสัย")
    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
    st.caption("🔍 การจำแนกธุรกรรมปกติและธุรกรรมทุจริตด้วยเทคนิคการเรียนรู้ของเครื่อง")

with h2:
    with st.container(border=True):
        pc1, pc2 = st.columns([1, 2], gap="medium")
        with pc1:
            if PHOTO:
                st.image(PHOTO, use_container_width=True)
            else:
                st.markdown('<div class="photo-placeholder">👤</div>', unsafe_allow_html=True)
        with pc2:
            st.markdown("**🧑‍💻 ผู้พัฒนาโปรเจกต์**")
            st.caption("นาย จตุรภัทร สถาปีตานนท์")
            st.caption("🆔 664245024 • 📚 66/43")
        if st.button("👤 ดูข้อมูลผู้พัฒนา", use_container_width=True):
            show_developer_info()
        st.markdown('<a class="dev-btn" href="https://github.com/aomaem21100-maker?tab=repositories" target="_blank">🔗 GitHub Profile</a>', unsafe_allow_html=True)

# ==================== METRICS ====================
st.markdown("")
m1, m2, m3, m4, _ = st.columns([1, 1, 1, 1, 1.2], gap="medium")
m1.metric("📊 ขนาดข้อมูล", "20,000")
m2.metric("🔢 คุณลักษณะ", "30 ตัว")
m3.metric("🏆 โมเดลที่ดีที่สุด", best["Model"] if best is not None else "–")
m4.metric("📈 F1-Score", f"{best['F1']:.2%}" if best is not None else "–")

st.markdown("")

# ==================== TABS ====================
t1, t2, t3, t4, t5 = st.tabs([
    "🎯 ปัญหาและข้อมูล", "🧹 Preprocessing", "🤖 โมเดล", "📊 การประเมินผล", "🔮 ทดลองตรวจจับ"
])

with t1:
    with st.container(border=True):
        st.subheader("🎯 1.1 การกำหนดปัญหา")
        st.write("💳 ธุรกรรมบัตรเครดิตที่ผิดปกติ (fraud) สร้างความเสียหายทางการเงินมหาศาล "
                 "แต่การตรวจสอบด้วยมนุษย์ทำได้ช้าและมีค่าใช้จ่ายสูง งานนี้จึงพัฒนาโมเดลการเรียนรู้ของเครื่องเพื่อตรวจจับ "
                 "pattern ของธุรกรรมที่น่าสงสัยแบบอัตโนมัติ")
    with st.container(border=True):
        st.subheader("📋 1.2 ชุดข้อมูล")
        st.write("📦 ข้อมูลธุรกรรมบัตรเครดิต 20,000 รายการ ประกอบด้วย 28 คุณลักษณะจาก PCA, "
                 "เวลา (Time), จำนวนเงิน (Amount) และตัวแปรเป้าหมาย Class (0=ปกติ, 1=fraud)")
        st.dataframe(make_data(10), use_container_width=True, hide_index=True)

with t2:
    with st.container(border=True):
        st.subheader("🧹 2.1 ขั้นตอนการเตรียมข้อมูล")
        st.markdown("""
        1️⃣ **การสุ่มตัวอย่าง** — ลดขนาดจาก 284,807 เป็น 20,000 รายการ แบบ Stratified

        2️⃣ **การจัดการ Imbalance** — ใช้ SMOTE เพื่อเพิ่มจำนวน fraud samples ในชุดฝึก

        3️⃣ **การปรับมาตราส่วน** — ใช้ StandardScaler กับ Amount และ Time

        4️⃣ **การแบ่งข้อมูล** — Train/Test = 80:20 แบบ Stratified

        5️⃣ **การเลือกเมตริก** — เน้น Precision, Recall, F1-Score
        """)

with t3:
    st.subheader("🤖 3.1 โมเดลที่ใช้ในการศึกษา")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("**🎯 Logistic Regression**")
            st.caption("📐 แบบจำลองเชิงเส้นสำหรับจำแนกไบนารี ใช้ class_weight='balanced'")
        with st.container(border=True):
            st.markdown("**🌲 Random Forest**")
            st.caption("🌳 Ensemble แบบ Bagging ลด variance พร้อม class_weight")
    with c2:
        with st.container(border=True):
            st.markdown("**🌳 Decision Tree**")
            st.caption("🔀 แบ่งกิ่งตามค่าที่ลด Gini Impurity ตีความง่าย")
        with st.container(border=True):
            st.markdown("**👥 K-NN**")
            st.caption("📏 จำแนกจาก k เพื่อนบ้านที่ใกล้ที่สุด ต้อง scaling ก่อน")

with t4:
    if comp is not None:
        with st.container(border=True):
            st.subheader("📊 4.1 ตารางเปรียบเทียบประสิทธิภาพ")
            st.dataframe(comp, use_container_width=True, hide_index=True)
    i1, i2 = st.columns(2, gap="medium")
    if os.path.exists("compare.png"):
        with i1:
            with st.container(border=True):
                st.image("compare.png", caption="📈 ภาพที่ 1: เปรียบเทียบประสิทธิภาพโมเดล", use_container_width=True)
    if os.path.exists("cm.png"):
        with i2:
            with st.container(border=True):
                st.image("cm.png", caption="🎯 ภาพที่ 2: Confusion Matrix", use_container_width=True)
    if os.path.exists("roc.png"):
        with st.container(border=True):
            st.image("roc.png", caption="📉 ภาพที่ 3: เส้นโค้ง ROC", use_container_width=True)
    if os.path.exists("pr_curve.png"):
        with st.container(border=True):
            st.image("pr_curve.png", caption="📊 ภาพที่ 4: เส้นโค้ง Precision-Recall", use_container_width=True)

with t5:
    with st.container(border=True):
        st.subheader("🔮 5.1 ทดลองตรวจจับธุรกรรม")

        if st.session_state.models is None:
            st.info("⏳ โมเดลยังไม่ถูกฝึก — กดปุ่มด้านล่างเพื่อเริ่มต้น")
            if st.button("🚀 เริ่มต้นฝึกโมเดล", use_container_width=True):
                try:
                    with st.spinner("⚙️ กำลังสร้างข้อมูลและฝึกโมเดล..."):
                        models, scaler = build_models()
                        st.session_state.models = models
                        st.session_state.scaler = scaler
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ ไม่สามารถฝึกโมเดลได้: {e}")
        else:
            models = st.session_state.models
            scaler = st.session_state.scaler
            model_name = st.selectbox("🎛️ เลือกโมเดล", list(models.keys()), index=2)

            st.markdown("**📝 กรอกข้อมูลธุรกรรม**")
            c1, c2 = st.columns(2)
            with c1:
                time_val = st.number_input("⏱️ Time (วินาที)", 0.0, 200000.0, value=50000.0)
                amount = st.number_input("💰 Amount (USD)", 0.0, 50000.0, value=100.0)
            with c2:
                v1 = st.number_input("🔢 V1 (PCA)", -50.0, 50.0, value=0.0)
                v2 = st.number_input("🔢 V2 (PCA)", -50.0, 50.0, value=0.0)

            if st.button("🔍 ตรวจจับธุรกรรม", use_container_width=True):
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

                st.markdown("")
                if pred == 1:
                    st.markdown(f"**🚨 ผลการตรวจจับ:** &nbsp; <span class='risk-fraud'>⚠️ ธุรกรรมน่าสงสัย (Fraud)</span>", unsafe_allow_html=True)
                    st.write(f"🎯 ความน่าจะเป็น fraud: **{proba:.1%}**")
                    st.progress(float(proba))
                else:
                    st.markdown(f"**✅ ผลการตรวจจับ:** &nbsp; <span class='risk-normal'>✓ ธุรกรรมปกติ (Normal)</span>", unsafe_allow_html=True)
                    st.write(f"🎯 ความน่าจะเป็น fraud: **{proba:.1%}**")
                    st.progress(float(proba))

# ==================== FOOTER ====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.caption("📚 จัดทำเพื่อประกอบการเรียนวิชา Machine Learning • 🛠️ พัฒนาด้วย Python, scikit-learn, Streamlit")
with footer_col2:
    st.markdown("<a href='https://github.com/aomaem21100-maker?tab=repositories' target='_blank' style='color:#3B5BDB; text-decoration:none;'>🔗 GitHub Repositories</a>", unsafe_allow_html=True)
with footer_col3:
    st.caption("© 2568 💙")