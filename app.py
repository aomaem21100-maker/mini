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

st.set_page_config(page_title="CKD Minimal Dashboard", page_icon="🩺", layout="wide")

# ==================== MINIMAL THEME #E0E0F3 ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
/* Palette: bg #E0E0F3 | card #FFFFFF | text #33335A | muted #8A8AA8 | accent #6C63FF */

html, body, [class*="css"] { font-family:'Prompt',sans-serif; }
div[data-testid="stAppViewContainer"], section.main, .stApp { background:#E0E0F3; }
#MainMenu, header, footer { visibility:hidden; }
h1,h2,h3,h4,p,span,li { color:#33335A; }

/* การ์ดขาวมุมมน */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#FFFFFF; border:none; border-radius:24px;
    box-shadow:0 4px 20px rgba(90,90,160,.07);
}
/* การ์ดตัวเลข */
div[data-testid="stMetric"] {
    background:#FFFFFF; border-radius:20px; padding:1.1rem 1.4rem;
    box-shadow:0 4px 20px rgba(90,90,160,.07);
}
div[data-testid="stMetric"] label { color:#8A8AA8 !important; font-weight:500; }

/* Tabs แบบยาเม็ด */
div[data-testid="stTabs"] ul { gap:.4rem; }
div[data-testid="stTabs"] button { background:transparent; color:#6E6E93; border-radius:999px; font-weight:500; }
div[data-testid="stTabs"] button[aria-selected="true"] { background:#6C63FF; color:#fff; }
div[data-testid="stTabs"] button:hover { color:#6C63FF; }

/* ปุ่ม */
div.stButton > button {
    background:#6C63FF; color:#fff; border:none; border-radius:14px;
    font-weight:600; padding:.55rem 2.5rem;
}
div.stButton > button:hover { background:#574FE0; }

/* ช่องกรอก */
input { border-radius:12px !important; border:1px solid #DCDCF0 !important; }
div[data-baseweb="select"] > div { border-radius:12px !important; border:1px solid #DCDCF0 !important; }
div[data-testid="stAlert"] { border-radius:16px; }
</style>
""", unsafe_allow_html=True)

# ==================== เทรนโมเดลบน Cloud (ไม่ใช้ .pkl) ====================
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
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oe", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols),
    ])
    models = {
        "Logistic Regression": Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000))]),
        "Decision Tree":       Pipeline([("pre", pre), ("m", DecisionTreeClassifier(random_state=42))]),
        "Random Forest":       Pipeline([("pre", pre), ("m", RandomForestClassifier(random_state=42))]),
        "K-NN":                Pipeline([("pre", pre), ("m", KNeighborsClassifier())]),
    }
    for p in models.values():
        p.fit(X, y)
    return X, models

X_ref, models = build_models()
comp = pd.read_csv("model_comparison.csv") if os.path.exists("model_comparison.csv") else None
best = comp.sort_values("Accuracy", ascending=False).iloc[0] if comp is not None else None

# ==================== HEADER ====================
h1, h2 = st.columns([3, 1], gap="large")
with h1:
    st.title("🩺 CKD Risk Dashboard")
    st.caption("ระบบคัดกรองความเสี่ยงโรคไตเรื้อรังด้วย Machine Learning")
with h2:
    with st.container(border=True):
        if os.path.exists("my_photo.jpg"):
            st.image("my_photo.jpg")
        st.markdown("**รหัส:** 63xxxxxxxx  \n**ชื่อ-นามสกุล:** ……………  \n**หมู่เรียน:** ……")

# ==================== METRICS ====================
m1, m2, m3, m4 = st.columns(4)
m1.metric("ข้อมูลตัวอย่าง", "400 แถว")
m2.metric("ฟีเจอร์", "24 ตัว")
m3.metric("โมเดลที่ดีที่สุด", best["Model"] if best is not None else "–")
m4.metric("Accuracy สูงสุด", f"{best['Accuracy']:.2%}" if best is not None else "–")

st.markdown("")

# ==================== TABS ====================
t1, t2, t3, t4, t5 = st.tabs(["📌 ปัญหา", "🧹 Preprocessing", "🤖 โมเดล", "📊 ประเมินผล", "🔮 ทำนาย"])

with t1:
    with st.container(border=True):
        st.subheader("การกำหนดปัญหา")
        st.write("โรคไตเรื้อรังระยะแรกมักไม่มีอาการชัดเจน → ใช้ ML คัดกรองผู้ป่วยเสี่ยงสูงจากค่าแล็บ เพื่อให้แพทย์วินิจฉัยได้เร็วขึ้น")
    with st.container(border=True):
        st.subheader("Dataset : Chronic Kidney Disease (UCI)")
        st.write("400 แถว • 24 คุณลักษณะ • ตัวแปรเป้าหมาย: classification (ckd / notckd)")
        st.dataframe(X_ref.head(8), use_container_width=True, hide_index=True)

with t2:
    with st.container(border=True):
        st.subheader("ขั้นตอน Data Preprocessing")
        st.markdown("""
        1. แก้ชนิดข้อมูล — แปลงคอลัมน์ pcv, wc, rc จากข้อความ → ตัวเลข
        2. จัดการค่าสูญหาย — ตัวเลขเติม Median / หมวดหมู่เติม Mode
        3. Encoding — แปลงข้อมูลหมวดหมู่เป็นตัวเลขด้วย Ordinal Encoding
        4. Scaling — ปรับสเกลด้วย StandardScaler (จำเป็นสำหรับ K-NN)
        5. Split — แบ่งข้อมูล Train/Test = 80/20 แบบ Stratified
        """)

with t3:
    mc1, mc2 = st.columns(2)
    with mc1.container(border=True):
        st.markdown("**Logistic Regression**  \nใช้ Sigmoid แปลงค่าเป็นความน่าจะเป็น 0–1 แล้วตัดที่ 0.5 เพื่อจำแนกคลาส")
    with mc2.container(border=True):
        st.markdown("**Decision Tree**  \nแบ่งข้อมูลเป็นกิ่งด้วยฟีเจอร์ที่ลดความไม่บริสุทธิ์ (Gini/Entropy) มากที่สุด")
    with mc1.container(border=True):
        st.markdown("**Random Forest**  \nสร้าง Decision Tree หลายต้นแบบ Bagging แล้วโหวตรวม ลด Overfitting")
    with mc2.container(border=True):
        st.markdown("**K-NN**  \nจำแนกจากเพื่อนบ้าน k ตัวที่ใกล้ที่สุดด้วยระยะทางยุคลิด จึงต้อง Scaling ก่อนเสมอ")

with t4:
    if comp is not None:
        with st.container(border=True):
            st.dataframe(comp, use_container_width=True, hide_index=True)
    i1, i2 = st.columns(2)
    if os.path.exists("compare.png"): i1.image("compare.png", caption="เปรียบเทียบโมเดล")
    if os.path.exists("roc.png"):     i2.image("roc.png", caption="ROC Curve")
    if os.path.exists("cm.png"):      st.image("cm.png", caption="Confusion Matrix ของโมเดลที่ดีที่สุด")

with t5:
    with st.container(border=True):
        st.subheader("ทดลองทำนาย")
        model_name = st.selectbox("เลือกโมเดล", list(models.keys()), index=2)
        st.caption("กรอกเฉพาะค่าหลัก 8 รายการ — ค่าที่เหลือระบบเติมมัธยฐาน/ฐานนิยมให้อัตโนมัติ")

        base_num = {c: float(X_ref[c].median()) for c in X_ref.select_dtypes(include="number").columns}
        base_cat = {c: X_ref[c].mode()[0] for c in X_ref.select_dtypes(include="object").columns}
        user_input = {**base_num, **base_cat}

        fa, fb = st.columns(2)
        with fa:
            user_input["age"]  = st.number_input("อายุ (ปี)", 1, 95, value=int(base_num["age"]))
            user_input["bp"]   = st.number_input("ความดันโลหิต (mmHg)", 50, 190, value=int(base_num["bp"]))
            user_input["bgr"]  = st.number_input("น้ำตาลในเลือด (bgr)", 20, 450, value=int(base_num["bgr"]))
            user_input["bu"]   = st.number_input("ยูเรีย (bu)", 1, 400, value=int(base_num["bu"]))
        with fb:
            user_input["sc"]    = st.number_input("ครีเอทินีน (sc)", 0.0, 80.0, value=float(base_num["sc"]))
            user_input["hemo"]  = st.number_input("ฮีโมโกลบิน (hemo)", 3.0, 18.0, value=float(base_num["hemo"]))
            user_input["sod"]   = st.number_input("โซเดียม (sod)", 50, 200, value=int(base_num["sod"]))
            user_input["k"]     = st.number_input("โพแทสเซียม (k)", 1.5, 8.0, value=float(base_num["k"]))

        if st.button("🔮 ทำนายผล", use_container_width=True):
            inp = pd.DataFrame([user_input])[X_ref.columns]
            m = models[model_name]
            pred = m.predict(inp)[0]
            prob = m.predict_proba(inp)[0][1] * 100
            if pred == 1:
                st.error(f"เสี่ยงโรคไตเรื้อรัง (ckd) • ความมั่นใจ {prob:.1f}%")
            else:
                st.success(f"ไม่เสี่ยงโรคไต (notckd) • ความมั่นใจ {100 - prob:.1f}%")import os
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

st.set_page_config(page_title="CKD Minimal Dashboard", page_icon="🩺", layout="wide")

# ==================== MINIMAL THEME #E0E0F3 ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
/* Palette: bg #E0E0F3 | card #FFFFFF | text #33335A | muted #8A8AA8 | accent #6C63FF */

html, body, [class*="css"] { font-family:'Prompt',sans-serif; }
div[data-testid="stAppViewContainer"], section.main, .stApp { background:#E0E0F3; }
#MainMenu, header, footer { visibility:hidden; }
h1,h2,h3,h4,p,span,li { color:#33335A; }

/* การ์ดขาวมุมมน */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#FFFFFF; border:none; border-radius:24px;
    box-shadow:0 4px 20px rgba(90,90,160,.07);
}
/* การ์ดตัวเลข */
div[data-testid="stMetric"] {
    background:#FFFFFF; border-radius:20px; padding:1.1rem 1.4rem;
    box-shadow:0 4px 20px rgba(90,90,160,.07);
}
div[data-testid="stMetric"] label { color:#8A8AA8 !important; font-weight:500; }

/* Tabs แบบยาเม็ด */
div[data-testid="stTabs"] ul { gap:.4rem; }
div[data-testid="stTabs"] button { background:transparent; color:#6E6E93; border-radius:999px; font-weight:500; }
div[data-testid="stTabs"] button[aria-selected="true"] { background:#6C63FF; color:#fff; }
div[data-testid="stTabs"] button:hover { color:#6C63FF; }

/* ปุ่ม */
div.stButton > button {
    background:#6C63FF; color:#fff; border:none; border-radius:14px;
    font-weight:600; padding:.55rem 2.5rem;
}
div.stButton > button:hover { background:#574FE0; }

/* ช่องกรอก */
input { border-radius:12px !important; border:1px solid #DCDCF0 !important; }
div[data-baseweb="select"] > div { border-radius:12px !important; border:1px solid #DCDCF0 !important; }
div[data-testid="stAlert"] { border-radius:16px; }
</style>
""", unsafe_allow_html=True)

# ==================== เทรนโมเดลบน Cloud (ไม่ใช้ .pkl) ====================
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
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oe", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols),
    ])
    models = {
        "Logistic Regression": Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000))]),
        "Decision Tree":       Pipeline([("pre", pre), ("m", DecisionTreeClassifier(random_state=42))]),
        "Random Forest":       Pipeline([("pre", pre), ("m", RandomForestClassifier(random_state=42))]),
        "K-NN":                Pipeline([("pre", pre), ("m", KNeighborsClassifier())]),
    }
    for p in models.values():
        p.fit(X, y)
    return X, models

X_ref, models = build_models()
comp = pd.read_csv("model_comparison.csv") if os.path.exists("model_comparison.csv") else None
best = comp.sort_values("Accuracy", ascending=False).iloc[0] if comp is not None else None

# ==================== HEADER ====================
h1, h2 = st.columns([3, 1], gap="large")
with h1:
    st.title("🩺 CKD Risk Dashboard")
    st.caption("ระบบคัดกรองความเสี่ยงโรคไตเรื้อรังด้วย Machine Learning")
with h2:
    with st.container(border=True):
        if os.path.exists("my_photo.jpg"):
            st.image("my_photo.jpg")
        st.markdown("**รหัส:** 63xxxxxxxx  \n**ชื่อ-นามสกุล:** ……………  \n**หมู่เรียน:** ……")

# ==================== METRICS ====================
m1, m2, m3, m4 = st.columns(4)
m1.metric("ข้อมูลตัวอย่าง", "400 แถว")
m2.metric("ฟีเจอร์", "24 ตัว")
m3.metric("โมเดลที่ดีที่สุด", best["Model"] if best is not None else "–")
m4.metric("Accuracy สูงสุด", f"{best['Accuracy']:.2%}" if best is not None else "–")

st.markdown("")

# ==================== TABS ====================
t1, t2, t3, t4, t5 = st.tabs(["📌 ปัญหา", "🧹 Preprocessing", "🤖 โมเดล", "📊 ประเมินผล", "🔮 ทำนาย"])

with t1:
    with st.container(border=True):
        st.subheader("การกำหนดปัญหา")
        st.write("โรคไตเรื้อรังระยะแรกมักไม่มีอาการชัดเจน → ใช้ ML คัดกรองผู้ป่วยเสี่ยงสูงจากค่าแล็บ เพื่อให้แพทย์วินิจฉัยได้เร็วขึ้น")
    with st.container(border=True):
        st.subheader("Dataset : Chronic Kidney Disease (UCI)")
        st.write("400 แถว • 24 คุณลักษณะ • ตัวแปรเป้าหมาย: classification (ckd / notckd)")
        st.dataframe(X_ref.head(8), use_container_width=True, hide_index=True)

with t2:
    with st.container(border=True):
        st.subheader("ขั้นตอน Data Preprocessing")
        st.markdown("""
        1. แก้ชนิดข้อมูล — แปลงคอลัมน์ pcv, wc, rc จากข้อความ → ตัวเลข
        2. จัดการค่าสูญหาย — ตัวเลขเติม Median / หมวดหมู่เติม Mode
        3. Encoding — แปลงข้อมูลหมวดหมู่เป็นตัวเลขด้วย Ordinal Encoding
        4. Scaling — ปรับสเกลด้วย StandardScaler (จำเป็นสำหรับ K-NN)
        5. Split — แบ่งข้อมูล Train/Test = 80/20 แบบ Stratified
        """)

with t3:
    mc1, mc2 = st.columns(2)
    with mc1.container(border=True):
        st.markdown("**Logistic Regression**  \nใช้ Sigmoid แปลงค่าเป็นความน่าจะเป็น 0–1 แล้วตัดที่ 0.5 เพื่อจำแนกคลาส")
    with mc2.container(border=True):
        st.markdown("**Decision Tree**  \nแบ่งข้อมูลเป็นกิ่งด้วยฟีเจอร์ที่ลดความไม่บริสุทธิ์ (Gini/Entropy) มากที่สุด")
    with mc1.container(border=True):
        st.markdown("**Random Forest**  \nสร้าง Decision Tree หลายต้นแบบ Bagging แล้วโหวตรวม ลด Overfitting")
    with mc2.container(border=True):
        st.markdown("**K-NN**  \nจำแนกจากเพื่อนบ้าน k ตัวที่ใกล้ที่สุดด้วยระยะทางยุคลิด จึงต้อง Scaling ก่อนเสมอ")

with t4:
    if comp is not None:
        with st.container(border=True):
            st.dataframe(comp, use_container_width=True, hide_index=True)
    i1, i2 = st.columns(2)
    if os.path.exists("compare.png"): i1.image("compare.png", caption="เปรียบเทียบโมเดล")
    if os.path.exists("roc.png"):     i2.image("roc.png", caption="ROC Curve")
    if os.path.exists("cm.png"):      st.image("cm.png", caption="Confusion Matrix ของโมเดลที่ดีที่สุด")

with t5:
    with st.container(border=True):
        st.subheader("ทดลองทำนาย")
        model_name = st.selectbox("เลือกโมเดล", list(models.keys()), index=2)
        st.caption("กรอกเฉพาะค่าหลัก 8 รายการ — ค่าที่เหลือระบบเติมมัธยฐาน/ฐานนิยมให้อัตโนมัติ")

        base_num = {c: float(X_ref[c].median()) for c in X_ref.select_dtypes(include="number").columns}
        base_cat = {c: X_ref[c].mode()[0] for c in X_ref.select_dtypes(include="object").columns}
        user_input = {**base_num, **base_cat}

        fa, fb = st.columns(2)
        with fa:
            user_input["age"]  = st.number_input("อายุ (ปี)", 1, 95, value=int(base_num["age"]))
            user_input["bp"]   = st.number_input("ความดันโลหิต (mmHg)", 50, 190, value=int(base_num["bp"]))
            user_input["bgr"]  = st.number_input("น้ำตาลในเลือด (bgr)", 20, 450, value=int(base_num["bgr"]))
            user_input["bu"]   = st.number_input("ยูเรีย (bu)", 1, 400, value=int(base_num["bu"]))
        with fb:
            user_input["sc"]    = st.number_input("ครีเอทินีน (sc)", 0.0, 80.0, value=float(base_num["sc"]))
            user_input["hemo"]  = st.number_input("ฮีโมโกลบิน (hemo)", 3.0, 18.0, value=float(base_num["hemo"]))
            user_input["sod"]   = st.number_input("โซเดียม (sod)", 50, 200, value=int(base_num["sod"]))
            user_input["k"]     = st.number_input("โพแทสเซียม (k)", 1.5, 8.0, value=float(base_num["k"]))

        if st.button("🔮 ทำนายผล", use_container_width=True):
            inp = pd.DataFrame([user_input])[X_ref.columns]
            m = models[model_name]
            pred = m.predict(inp)[0]
            prob = m.predict_proba(inp)[0][1] * 100
            if pred == 1:
                st.error(f"เสี่ยงโรคไตเรื้อรัง (ckd) • ความมั่นใจ {prob:.1f}%")
            else:
                st.success(f"ไม่เสี่ยงโรคไต (notckd) • ความมั่นใจ {100 - prob:.1f}%")