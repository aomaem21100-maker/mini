import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Machine Learning Hub", page_icon="🤖", layout="wide")
st.session_state.setdefault("models", None)
st.session_state.setdefault("scaler", None)

# ==================== CYBER DARK THEME ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=Orbitron:wght@500;700&family=Inter:wght@400;600;700&display=swap');

/* ===== พื้นหลังหลัก + ลายกริด ===== */
div[data-testid="stAppViewContainer"] > section.main {
    background-color: #0A101C;
    background-image:
        linear-gradient(rgba(45,224,200,0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(45,224,200,0.045) 1px, transparent 1px);
    background-size: 42px 42px;
}
#MainMenu, header, footer { visibility: hidden; }

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: #0D1526;
    border-right: 1px solid #16233C;
}
.hub-title {
    font-family: 'Orbitron', 'IBM Plex Sans Thai', sans-serif;
    background: linear-gradient(90deg, #2DE0C8, #8B5CF6);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    letter-spacing: 3px; font-weight: 700; font-size: 1.05rem;
}
.hub-hr { border: none; height: 1px; background: #1C2B47; margin: .9rem 0 1.2rem 0; }

/* ===== เมนูนำทาง (radio → nav item) ===== */
section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: block; padding: .75rem 1rem; border-radius: 10px;
    color: #8FA3C4; font-weight: 500; cursor: pointer;
    border-left: 3px solid transparent; margin-bottom: .35rem;
    transition: all .2s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { color: #E6EDF7; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(23,195,178,.28), rgba(123,47,247,.28));
    border-left: 3px solid #2DE0C8;
    color: #FFFFFF;
}

/* ===== ข้อความ ===== */
html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', 'Inter', sans-serif; }
h1,h2,h3,h4 { color: #E6EDF7; font-weight: 600; }
p, li { color: #C7D3E8; }
caption, small { color: #7C8DB0 !important; }

.grad-title {
    font-size: 2.3rem; font-weight: 800;
    background: linear-gradient(90deg, #2DE0C8 0%, #4CC9F0 45%, #8B5CF6 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.tag-cyber {
    display: inline-block; font-family: 'Orbitron', sans-serif;
    font-size: .68rem; letter-spacing: 2.5px; color: #2DE0C8;
    border: 1px solid rgba(45,224,200,.5); border-radius: 4px;
    padding: .25rem .7rem; margin-bottom: .6rem;
}

/* ===== การ์ด ===== */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #0F1930; border: 1px solid #1C2B47; border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,.35);
}

/* ===== Metrics ===== */
div[data-testid="stMetric"] {
    background: #0F1930; border: 1px solid #1C2B47;
    border-left: 3px solid #2DE0C8; border-radius: 12px; padding: 1rem 1.2rem;
}
div[data-testid="stMetric"] label { color: #7C8DB0 !important; font-size: .78rem; font-weight: 600; letter-spacing: .5px; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #2DE0C8; font-weight: 700; }

/* ===== Tabs ===== */
div[data-testid="stTabs"] ul { gap: 0; border-bottom: 1px solid #1C2B47; background: transparent; }
div[data-testid="stTabs"] button {
    background: transparent; color: #7C8DB0; border-radius: 0;
    font-weight: 500; padding: .7rem 1.3rem; border: none;
    border-bottom: 3px solid transparent;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2DE0C8; border-bottom: 3px solid #2DE0C8;
    font-weight: 600; background: transparent; box-shadow: none;
}
div[data-testid="stTabs"] button:hover { color: #2DE0C8; background: transparent; }

/* ===== ปุ่ม gradient ===== */
div.stButton > button {
    background: linear-gradient(90deg, #17C3B2, #7B2FF7);
    color: #FFFFFF; border: none; border-radius: 10px;
    font-weight: 600; padding: .55rem 1.8rem;
    box-shadow: 0 2px 10px rgba(23,195,178,.25);
    transition: all .2s ease;
}
div.stButton > button:hover { filter: brightness(1.15); transform: translateY(-1px); }

/* ===== Inputs ===== */
input, textarea {
    background: #0F1930 !important; border: 1px solid #24344F !important;
    color: #E6EDF7 !important; border-radius: 8px !important;
}
input:focus, textarea:focus { border-color: #2DE0C8 !important; box-shadow: 0 0 0 3px rgba(45,224,200,.12) !important; }
div[data-baseweb="select"] > div { background: #0F1930 !important; border: 1px solid #24344F !important; color: #E6EDF7 !important; }

/* ===== Progress / Alert / Expander ===== */
div[data-testid="stProgress"] > div { background: #16233C; border-radius: 6px; }
div[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #17C3B2, #7B2FF7); border-radius: 6px; }
div[data-testid="stAlert"] { background: #0F1930; border: 1px solid #1C2B47; border-radius: 12px; color: #C7D3E8; }
div[data-testid="stExpander"] { background: #0F1930; border: 1px solid #1C2B47; border-radius: 12px; }

hr { border-color: #1C2B47 !important; }
img { border-radius: 12px; border: 1px solid #1C2B47; }

/* ===== Developer Page ===== */
.info-row { display: flex; gap: .6rem; padding: .4rem 0; font-size: .95rem; }
.info-row .label { color: #7C8DB0; min-width: 140px; }
.info-row .value { color: #E6EDF7; font-weight: 600; }
.tech-chip {
    display: inline-block; margin: .2rem .35rem .2rem 0; padding: .4rem .9rem;
    border-radius: 999px; border: 1px solid rgba(45,224,200,.6);
    color: #2DE0C8; font-size: .82rem; font-weight: 600;
}
.avatar-box {
    width: 110px; height: 110px; border-radius: 50%;
    background: linear-gradient(135deg, rgba(23,195,178,.25), rgba(123,47,247,.25));
    border: 2px solid #2DE0C8;
    display: flex; align-items: center; justify-content: center;
    font-size: 3rem;
}
.cyber-link { color: #2DE0C8; text-decoration: none; font-weight: 600; }
.cyber-link:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ==================== Helpers ====================
def find_photo():
    try:
        files = os.listdir(".")
    except Exception:
        return None
    preferred = ["my_photo.jpg", "my_photo.png", "photo.jpg", "photo.png", "profile.jpg", "profile.png"]
    for name in preferred:
        if name in files:
            return name
    exclude = ("compare", "cm", "roc", "pr_curve", "confusion", "icon", "logo")
    for f in sorted(files):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and not any(k in f.lower() for k in exclude):
            return f
    return None

PHOTO = find_photo()

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

# ==================== SIDEBAR NAV ====================
st.sidebar.markdown('<div class="hub-title">MACHINE LEARNING HUB</div>', unsafe_allow_html=True)
st.sidebar.markdown('<hr class="hub-hr">', unsafe_allow_html=True)
page = st.sidebar.radio("นำทาง", ["หน้าหลัก", "ผู้พัฒนา"], label_visibility="collapsed")

# ================================================================
#                         หน้าหลัก
# ================================================================
if page == "หน้าหลัก":
    st.markdown('<span class="tag-cyber">🤖 MACHINE LEARNING PROJECT</span>', unsafe_allow_html=True)
    st.markdown('<div class="grad-title">ระบบตรวจจับธุรกรรมที่น่าสงสัย</div>', unsafe_allow_html=True)
    st.caption("🔍 การจำแนกธุรกรรมปกติและธุรกรรมทุจริตด้วยเทคนิคการเรียนรู้ของเครื่อง")

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 ขนาดข้อมูล", "20,000 รายการ")
    m2.metric("🔢 คุณลักษณะ", "30 ตัว")
    m3.metric("🏆 โมเดลที่ดีที่สุด", best["Model"] if best is not None else "–")
    m4.metric("📈 F1-Score", f"{best['F1']:.2%}" if best is not None else "–")

    st.markdown("")

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
        c1, c2 = st.columns(2)
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
                st.markdown("**👥 K-Nearest Neighbors (K-NN)**")
                st.caption("📏 จำแนกจาก k เพื่อนบ้านที่ใกล้ที่สุด ต้อง scaling ก่อน")

    with t4:
        if comp is not None:
            with st.container(border=True):
                st.subheader("📊 4.1 ตารางเปรียบเทียบประสิทธิภาพ")
                st.dataframe(comp, use_container_width=True, hide_index=True)
        i1, i2 = st.columns(2)
        if os.path.exists("compare.png"):
            with i1:
                with st.container(border=True):
                    st.image("compare.png", caption="📈 กราฟเปรียบเทียบประสิทธิภาพโมเดล", use_container_width=True)
        if os.path.exists("cm.png"):
            with i2:
                with st.container(border=True):
                    st.image("cm.png", caption="🎯 Confusion Matrix ของโมเดลที่ดีที่สุด", use_container_width=True)
        if os.path.exists("roc.png"):
            with st.container(border=True):
                st.image("roc.png", caption="📉 เส้นโค้ง ROC", use_container_width=True)
        if os.path.exists("pr_curve.png"):
            with st.container(border=True):
                st.image("pr_curve.png", caption="📊 เส้นโค้ง Precision-Recall", use_container_width=True)

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

                    st.markdown("---")
                    if pred == 1:
                        st.error(f"🚨 **ผลการตรวจจับ:** ธุรกรรมน่าสงสัย (Fraud)")
                        st.write(f"🎯 ความน่าจะเป็น fraud: {proba:.1%}")
                        st.progress(float(proba))
                    else:
                        st.success(f"✅ **ผลการตรวจจับ:** ธุรกรรมปกติ (Normal)")
                        st.write(f"🎯 ความน่าจะเป็น fraud: {proba:.1%}")
                        st.progress(float(proba))

    st.markdown("---")
    st.caption("📚 จัดทำเพื่อประกอบการเรียนวิชา Machine Learning • 🛠️ พัฒนาด้วย Python, scikit-learn, Streamlit")

# ================================================================
#                         ผู้พัฒนา
# ================================================================
else:
    st.markdown('<span class="tag-cyber">👤 DEVELOPER</span>', unsafe_allow_html=True)
    st.markdown('<div class="grad-title">ผู้พัฒนาโปรเจกต์</div>', unsafe_allow_html=True)
    st.markdown("")

    with st.container(border=True):
        a1, a2 = st.columns([1, 2], gap="large")
        with a1:
            if PHOTO:
                st.image(PHOTO, use_container_width=True)
            else:
                st.markdown('<div class="avatar-box">👤</div>', unsafe_allow_html=True)
        with a2:
            st.markdown("""
            <div class="info-row"><span class="label">👤 ชื่อ-นามสกุล</span><span class="value">นาย จตุรภัทร สถาปีตานนท์</span></div>
            <div class="info-row"><span class="label">🆔 รหัสนักศึกษา</span><span class="value">664245024</span></div>
            <div class="info-row"><span class="label">📚 หมู่เรียน</span><span class="value">66/43</span></div>
            <div class="info-row"><span class="label">🎓 สาขา</span><span class="value">วิทยาการคอมพิวเตอร์</span></div>
            """, unsafe_allow_html=True)

    st.markdown("")
    with st.container(border=True):
        st.subheader("📝 เกี่ยวกับโปรเจกต์")
        st.write("💡 โครงงานนี้เป็นส่วนหนึ่งของวิชา Machine Learning โดยมีวัตถุประสงค์เพื่อศึกษาและประยุกต์ใช้"
                 "เทคนิคการเรียนรู้ของเครื่องในการตรวจจับธุรกรรมบัตรเครดิตที่น่าสงสัย")

        st.subheader("🛠️ เทคโนโลยีที่ใช้")
        st.markdown("""
        <span class="tech-chip">🐍 Python</span>
        <span class="tech-chip">🤖 scikit-learn</span>
        <span class="tech-chip">🚀 Streamlit</span>
        <span class="tech-chip">⚖️ imbalanced-learn</span>
        <span class="tech-chip">🐼 pandas</span>
        <span class="tech-chip">🔢 NumPy</span>
        """, unsafe_allow_html=True)

        st.subheader("🔗 ลิงก์ที่เกี่ยวข้อง")
        st.markdown("- 🌐 [GitHub Profile](https://github.com/aomaem21100-maker?tab=repositories)\n"
                    "- 📦 [Source Code โปรเจกต์นี้](https://github.com/aomaem21100-maker)\n"
                    "- 📚 [Dataset: Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)")

    st.markdown("---")
    st.caption("© 2568 • Machine Learning Hub 💙")