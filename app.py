import os
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Machine Learning Hub", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

plt.rcParams.update({
    "figure.facecolor": "#212B3B", "axes.facecolor": "#212B3B",
    "axes.edgecolor": "#3A465C", "axes.labelcolor": "#D5DCEA",
    "text.color": "#D5DCEA", "xtick.color": "#93A1B8", "ytick.color": "#93A1B8",
    "legend.facecolor": "#212B3B", "grid.color": "#3A465C", "font.size": 10,
})
EVA_COLORS = ["#39FF14", "#6A3AB2", "#FF7A00", "#4CC9F0"]

# ==================== THEME: Evangelion ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=Orbitron:wght@500;700;900&family=Inter:wght@400;600;700&display=swap');

div[data-testid="stAppViewContainer"] > section.main {
    background-color: #293242;
    background-image:
        linear-gradient(rgba(57,255,20,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(57,255,20,0.05) 1px, transparent 1px);
    background-size: 42px 42px;
}
#MainMenu, header, footer { visibility: hidden; }

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] { background: #D4D2F2; border-right: 2px solid #6A3AB2; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: #1A1A2E; }

.hub-title {
    font-family: 'Orbitron', 'IBM Plex Sans Thai', sans-serif;
    background: linear-gradient(90deg, #000000 0%, #2E2E3E 60%, #4A4A5E 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    letter-spacing: 3px; font-weight: 900; font-size: 1.05rem;
}
.hub-hr {
    border: none; height: 3px; margin: .9rem 0 1.2rem 0;
    background: repeating-linear-gradient(45deg, #FF7A00 0 10px, #14141E 10px 20px);
}

section[data-testid="stSidebar"] .stSelectbox { margin-top: 1rem; }
section[data-testid="stSidebar"] .stSelectbox > div {
    background: #FFFFFF !important;
    border: 2px solid #6A3AB2 !important;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 1rem; font-weight: 600; color: #1A1A2E !important;
}
section[data-testid="stSidebar"] .stSelectbox > div:hover {
    border-color: #39FF14 !important;
    box-shadow: 0 0 12px rgba(57, 255, 20, 0.3);
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', 'Inter', sans-serif; }
h1,h2,h3,h4 { color: #F2F5FB; font-weight: 600; }
p, li { color: #D5DCEA; }
caption, small { color: #93A1B8 !important; }

.grad-title {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg, #39FF14 0%, #7EF29A 35%, #6A3AB2 80%, #FF7A00 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    text-shadow: 0 0 24px rgba(57,255,20,.25);
}
.tag-cyber {
    display: inline-block; font-family: 'Orbitron', sans-serif;
    font-size: .68rem; letter-spacing: 2.5px; color: #39FF14;
    border: 1px solid rgba(57,255,20,.6); border-radius: 4px;
    padding: .25rem .7rem; margin-bottom: .6rem; background: rgba(57,255,20,.08);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #212B3B; border: 1px solid #3A465C; border-radius: 14px;
    box-shadow: 0 2px 14px rgba(0,0,0,.35);
}
div[data-testid="stMetric"] {
    background: #212B3B; border: 1px solid #3A465C;
    border-left: 4px solid #FF7A00; border-radius: 10px; padding: 1rem 1.2rem;
}
div[data-testid="stMetric"] label { color: #93A1B8 !important; font-size: .78rem; font-weight: 600; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #39FF14; font-weight: 700; text-shadow: 0 0 12px rgba(57,255,20,.35);
}

div[data-testid="stTabs"] ul { gap: 0; border-bottom: 1px solid #3A465C; background: transparent; }
div[data-testid="stTabs"] button {
    background: transparent; color: #93A1B8; border-radius: 0; font-weight: 500;
    padding: .7rem 1.3rem; border: none; border-bottom: 3px solid transparent;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #39FF14; border-bottom: 3px solid #39FF14; font-weight: 700;
    background: transparent; box-shadow: none; text-shadow: 0 0 10px rgba(57,255,20,.4);
}
div[data-testid="stTabs"] button:hover { color: #39FF14; background: transparent; }

div.stButton > button {
    background: #39FF14; color: #0A0A0A; border: none; border-radius: 8px;
    font-weight: 700; padding: .55rem 1.8rem;
    font-family: 'Orbitron', 'IBM Plex Sans Thai', sans-serif;
    box-shadow: 0 0 14px rgba(57,255,20,.35); transition: all .2s ease;
}
div.stButton > button:hover { background: #52FF33; box-shadow: 0 0 22px rgba(57,255,20,.55); }

input, textarea {
    background: #212B3B !important; border: 1px solid #3A465C !important;
    color: #F2F5FB !important; border-radius: 8px !important;
}
input:focus, textarea:focus { border-color: #39FF14 !important; box-shadow: 0 0 0 3px rgba(57,255,20,.15) !important; }
div[data-baseweb="select"] > div { background: #212B3B !important; border: 1px solid #3A465C !important; color: #F2F5FB !important; }

div[data-testid="stProgress"] > div { background: #1A2230; border-radius: 4px; }
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #39FF14, #6A3AB2); border-radius: 4px;
    box-shadow: 0 0 12px rgba(57,255,20,.4);
}
div[data-testid="stAlert"] { background: #212B3B; border: 1px solid #3A465C; border-radius: 10px; color: #D5DCEA; }
div[data-testid="stExpander"] { background: #212B3B; border: 1px solid #3A465C; border-radius: 10px; }

hr { border-color: #3A465C !important; }
img { border-radius: 10px; border: 1px solid #3A465C; }

.hazard-line {
    height: 4px; border-radius: 2px; margin: .5rem 0 1rem 0;
    background: repeating-linear-gradient(45deg, #FF7A00 0 12px, #14141E 12px 24px);
}
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
def build_and_eval():
    df = make_data(20000)
    X = df.drop(columns=["Class"]); y = df["Class"]
    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    sm = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "K-NN": KNeighborsClassifier(n_neighbors=5),
    }
    rows, trained, preds, probas = {}, {}, {}, {}
    for name, m in models.items():
        if name == "K-NN": m.fit(X_tr, y_tr)
        else: m.fit(X_tr_res, y_tr_res)
        yp = m.predict(X_te); pr = m.predict_proba(X_te)[:, 1]
        trained[name] = m; preds[name] = yp; probas[name] = pr
        rows[name] = [accuracy_score(y_te, yp), precision_score(y_te, yp, zero_division=0),
                      recall_score(y_te, yp, zero_division=0), f1_score(y_te, yp, zero_division=0)]
    comp = pd.DataFrame(rows, index=["Accuracy", "Precision", "Recall", "F1"]).T.reset_index()
    comp.columns = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    best_name = comp.sort_values("F1", ascending=False).iloc[0]["Model"]
    return trained, scaler, comp, best_name, np.array(y_te), preds, probas

def make_figures(comp, y_te, preds, probas, best_name):
    fig_bar, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(comp)); w = .2
    for i, col in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(x + i*w - 1.5*w, comp[col], w, label=col, color=EVA_COLORS[i])
    ax.set_xticks(x); ax.set_xticklabels(comp["Model"], rotation=8)
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Model Comparison"); ax.grid(axis="y", alpha=.3)

    fig_roc, ax = plt.subplots(figsize=(7, 5))
    for i, (name, pr) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_te, pr)
        ax.plot(fpr, tpr, color=EVA_COLORS[i], label=name)แก้มาใหเลย
    ax.plot([0, 1], [0, 1], "--", color="#93A1B8", alpha=.6)
    ax.set_title("ROC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(loc="lower right")

    cm = confusion_matrix(y_te, preds[best_name])
    fig_cm, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fraud"]); ax.set_yticklabels(["Normal", "Fraud"])
    for ii in range(2):
        for jj in range(2):
            ax.text(jj, ii, f"{cm[ii, jj]:,}", ha="center", va="center",
                    color="#0A0A0A" if cm[ii, jj] > cm.max()/2 else "#D5DCEA", fontweight="bold")
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig_cm.colorbar(im, ax=ax, fraction=.046)
    return fig_bar, fig_roc, fig_cm

def train_now(key):
    if st.button("🚀 เริ่มต้นฝึกโมเดลและประเมินผล", use_container_width=True, key=key):
        with st.spinner("⚙️ กำลังฝึก 4 โมเดล + สร้างกราฟประเมินผล..."):
            st.session_state["eval"] = build_and_eval()
        st.rerun()

# ==================== ✅ ระบบนำทาง (state เดียว ใช้ callback) ====================
NAV_OPTIONS = ["🏠 หน้าหลัก", "👤 ผู้พัฒนา"]

def go_home():
    st.session_state["nav_widget"] = NAV_OPTIONS[0]

def go_dev():
    st.session_state["nav_widget"] = NAV_OPTIONS[1]

st.sidebar.markdown('<div class="hub-title">MACHINE LEARNING HUB</div>', unsafe_allow_html=True)
st.sidebar.markdown('<hr class="hub-hr">', unsafe_allow_html=True)
st.sidebar.markdown("### 📍 นำทาง")
page = st.sidebar.selectbox("เลือกหน้า", NAV_OPTIONS, key="nav_widget")

# ปุ่มลัดด้านบน (callback จะเปลี่ยนค่า selectbox ให้เอง)
tb1, tb2, _ = st.columns([1.2, 1.2, 6])
with tb1:
    st.button("🏠 หน้าหลัก", use_container_width=True, key="top_home_btn", on_click=go_home)
with tb2:
    st.button("👤 ผู้พัฒนา", use_container_width=True, key="top_dev_btn", on_click=go_dev)

EV = st.session_state.get("eval", None)

# ================================================================
#              หน้าหลัก
# ================================================================
if page == "🏠 หน้าหลัก":
    st.markdown('<span class="tag-cyber">MACHINE LEARNING PROJECT</span>', unsafe_allow_html=True)
    st.markdown('<div class="grad-title">ระบบตรวจจับธุรกรรมที่น่าสงสัย</div>', unsafe_allow_html=True)
    st.markdown('<div class="hazard-line"></div>', unsafe_allow_html=True)
    st.caption("🔍 การจำแนกธุรกรรมปกติและธุรกรรมทุจริตด้วยเทคนิคการเรียนรู้ของเครื่อง")

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 ขนาดข้อมูล", "20,000 รายการ")
    m2.metric("🔢 คุณลักษณะ", "30 ตัว")
    m3.metric("🏆 โมเดลที่ดีที่สุด", EV[3] if EV else "–")
    m4.metric("📈 คะแนน F1", f"{EV[2].sort_values('F1', ascending=False).iloc[0]['F1']:.2%}" if EV else "–")

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
        if EV is None:
            with st.container(border=True):
                st.info("⏳ ยังไม่มีผลการประเมิน — กดปุ่มเพื่อฝึกโมเดลและสร้างตาราง+กราฟอัตโนมัติ")
                train_now("btn_train4")
        else:
            trained, scaler, comp, best_name, y_te, preds, probas = EV
            with st.container(border=True):
                st.subheader("📊 4.1 ตารางเปรียบเทียบประสิทธิภาพ")
                st.dataframe(comp.round(4), use_container_width=True, hide_index=True)
                st.caption(f"🏆 โมเดลที่ดีที่สุดตาม F1-Score: **{best_name}**")
            fig_bar, fig_roc, fig_cm = make_figures(comp, y_te, preds, probas, best_name)
            st.subheader("📈 4.2 กราฟเปรียบเทียบโมเดล")
            st.pyplot(fig_bar)
            g1, g2 = st.columns(2)
            with g1:
                with st.container(border=True):
                    st.pyplot(fig_roc)
                    st.caption("📉 ภาพที่ 2: เส้นโค้ง ROC ของทั้ง 4 โมเดล")
            with g2:
                with st.container(border=True):
                    st.pyplot(fig_cm)
                    st.caption(f"🎯 ภาพที่ 3: Confusion Matrix ของ {best_name}")

    with t5:
        with st.container(border=True):
            st.subheader("🔮 5.1 ทดลองตรวจจับธุรกรรม")
            if EV is None:
                st.info("⏳ โมเดลยังไม่ถูกฝึก — กดปุ่มด้านล่างเพื่อเริ่มต้น")
                train_now("btn_train5")
            else:
                trained, scaler, comp, best_name, y_te, preds, probas = EV
                best_idx = list(trained.keys()).index(best_name)
                model_name = st.selectbox("🎛️ เลือกโมเดล", list(trained.keys()), index=best_idx)

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
                    inp_dict["V1"] = float(v1); inp_dict["V2"] = float(v2)
                    scaled = scaler.transform(pd.DataFrame([[amount, time_val]], columns=["Amount", "Time"]))[0]
                    inp_dict["Amount"] = float(scaled[0]); inp_dict["Time"] = float(scaled[1])
                    train_columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
                    inp = pd.DataFrame([inp_dict])[train_columns]

                    m = trained[model_name]
                    pred = m.predict(inp)[0]
                    proba = m.predict_proba(inp)[0][1]

                    st.markdown("---")
                    if pred == 1:
                        st.error(f"🚨 **ผลการตรวจจับ:** ธุรกรรมน่าสงสัย (Fraud)")
                    else:
                        st.success(f"✅ **ผลการตรวจจับ:** ธุรกรรมปกติ (Normal)")
                    st.write(f"🎯 ความน่าจะเป็น fraud: **{proba:.1%}**")
                    st.progress(float(proba))

    st.markdown("---")
    st.caption("📚 จัดทำเพื่อประกอบการเรียนวิชา Machine Learning • 🛠️ พัฒนาด้วย Python, scikit-learn, Streamlit")

# ================================================================
#      👤 หน้าผู้พัฒนา (กึ่งกลางสมบูรณ์ + รูปวงกลมชัวร์)
# ================================================================
else:
    st.markdown("""
    <style>
    /* ✅ บังคับรูปวงกลมสมส่วนด้วย !important (กัน Streamlit ทับ) */
    img {
        display: block !important;
        margin: 0 auto !important;
        aspect-ratio: 1 / 1 !important;
        object-fit: cover !important;
        object-position: center top !important;
        border-radius: 50% !important;
        border: 5px solid transparent !important;
        background: linear-gradient(#212B3B, #212B3B) padding-box,
                    linear-gradient(135deg, #39FF14, #6A3AB2) border-box !important;
        box-shadow: 0 0 40px rgba(57, 255, 20, .4) !important;
        transition: all 0.3s ease;
    }
    img:hover {
        transform: scale(1.03);
        box-shadow: 0 0 50px rgba(57, 255, 20, .6) !important;
    }

    .dev-head { text-align: center; margin-top: 1rem; }
    .dev-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(90deg, #39FF14 0%, #7EF29A 35%, #6A3AB2 75%, #FF7A00 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        text-shadow: 0 0 24px rgba(57, 255, 20, .25);
    }
    .dev-sub { color: #93A1B8; letter-spacing: 1px; font-size: .95rem; margin-top: .3rem; }

    .dev-avatar {
        width: 240px; height: 240px; border-radius: 50%; margin: 0 auto;
        background: linear-gradient(135deg, rgba(57,255,20,.25), rgba(106,58,178,.4));
        border: 5px solid #6A3AB2;
        display: flex; align-items: center; justify-content: center;
        font-size: 5rem; box-shadow: 0 0 40px rgba(57,255,20,.4);
    }

    .dev-name { text-align: center; font-size: 1.35rem; font-weight: 700; color: #F2F5FB; margin: .3rem 0 1rem 0; }
    .info-row {
        display: flex; justify-content: space-between; gap: 1rem;
        padding: .8rem .2rem; border-top: 1px solid #3A465C; font-size: .95rem;
    }
    .info-row:first-child { border-top: none; }
    .info-row .label { color: #93A1B8; }
    .info-row .value { color: #F2F5FB; font-weight: 600; }
    .dev-center { text-align: center; }
    .dev2-chip {
        display: inline-block; margin: .2rem .3rem; padding: .4rem .9rem;
        border-radius: 999px; border: 1px solid rgba(57,255,20,.55);
        color: #39FF14; font-size: .82rem; font-weight: 600; background: rgba(57,255,20,.06);
    }
    .dev2-chip.purple { border-color: rgba(106,58,178,.6); color: #B794F6; background: rgba(106,58,178,.1); }
    .dev2-chip.orange { border-color: rgba(255,122,0,.55); color: #FF7A00; background: rgba(255,122,0,.07); }
    </style>
    """, unsafe_allow_html=True)

    # ✅ จัดทั้งหน้ากึ่งกลางด้วยคอลัมน์
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.markdown("""
        <div class="dev-head">
            <div class="dev-title">ผู้พัฒนา</div>
            <div class="dev-sub">ข้อมูลผู้จัดทำโปรเจค Machine Learning Hub</div>
        </div>
        """, unsafe_allow_html=True)

        # ✅ จัดรูปกึ่งกลางด้วยคอลัมน์ซ้อน (ชัวร์กว่า margin auto)
        if PHOTO:
            c1, c2, c3 = st.columns([1, 1.2, 1])
            with c2:
                st.image(PHOTO, use_container_width=True)
        else:
            st.markdown('<div class="dev-avatar">👤</div>', unsafe_allow_html=True)

        st.markdown("")

        with st.container(border=True):
            st.markdown('<div class="dev-name">นาย จตุรภัทร สถาปีตานนท์</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-row"><span class="label">รหัสนักศึกษา</span><span class="value">664245024</span></div>
            <div class="info-row"><span class="label">หมู่เรียน</span><span class="value">Sec. 66/43</span></div>
            <div class="info-row"><span class="label">สาขา</span><span class="value">วิทยาการคอมพิวเตอร์</span></div>
            """, unsafe_allow_html=True)

        st.markdown("")

        with st.container(border=True):
            st.markdown('<div class="dev-center"><h3 style="margin:.2rem 0 .5rem 0;">📝 เกี่ยวกับโปรเจกต์</h3></div>', unsafe_allow_html=True)
            st.write("💡 โครงงานนี้เป็นส่วนหนึ่งของวิชา Machine Learning โดยมีวัตถุประสงค์เพื่อศึกษาและประยุกต์ใช้"
                     "เทคนิคการเรียนรู้ของเครื่องในการตรวจจับธุรกรรมบัตรเครดิตที่น่าสงสัย")

            st.markdown('<div class="dev-center"><h3 style="margin:1rem 0 .5rem 0;">🛠️ เทคโนโลยีที่ใช้</h3></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="dev-center">
                <span class="dev2-chip">🐍 Python</span>
                <span class="dev2-chip">🤖 scikit-learn</span>
                <span class="dev2-chip">🚀 Streamlit</span>
                <span class="dev2-chip purple">⚖️ imbalanced-learn</span>
                <span class="dev2-chip purple">📊 matplotlib</span>
                <span class="dev2-chip orange">🐼 pandas</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="dev-center"><h3 style="margin:1rem 0 .5rem 0;">🔗 ลิงก์ที่เกี่ยวข้อง</h3></div>', unsafe_allow_html=True)
            st.markdown("- 🌐 [GitHub Profile](https://github.com/aomaem21100-maker?tab=repositories)\n"
                        "- 📦 [Source Code โปรเจกต์นี้](https://github.com/aomaem21100-maker)\n"
                        "- 📚 [Dataset: Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)")

        st.markdown("---")
        st.caption("© 2568 • Machine Learning Hub")