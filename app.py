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
                             f1_score, roc_curve, confusion_matrix, 
                             classification_report, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, VotingClassifier, 
                              IsolationForest, ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Machine Learning Hub - Advanced", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

plt.rcParams.update({
    "figure.facecolor": "#212B3B", "axes.facecolor": "#212B3B",
    "axes.edgecolor": "#3A465C", "axes.labelcolor": "#D5DCEA",
    "text.color": "#D5DCEA", "xtick.color": "#93A1B8", "ytick.color": "#93A1B8",
    "legend.facecolor": "#212B3B", "grid.color": "#3A465C", "font.size": 10,
})
EVA_COLORS = ["#39FF14", "#6A3AB2", "#FF7A00", "#4CC9F0", "#FF3232", "#B794F6", "#FFD700", "#00CED1"]

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

.hazard-line {
    height: 4px; border-radius: 2px; margin: .5rem 0 1rem 0;
    background: repeating-linear-gradient(45deg, #FF7A00 0 12px, #14141E 12px 24px);
}

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

.model-card {
    background: linear-gradient(135deg, rgba(57,255,20,0.1) 0%, rgba(106,58,178,0.1) 100%);
    border: 1px solid #3A465C; border-radius: 12px; padding: 1rem;
    margin: 0.5rem 0; transition: all 0.3s ease;
}
.model-card:hover {
    border-color: #39FF14; box-shadow: 0 0 20px rgba(57,255,20,0.2);
}

.risk-badge {
    display: inline-block; padding: 0.4rem 1rem; border-radius: 6px;
    font-family: 'Orbitron', sans-serif; font-weight: 700; letter-spacing: 1px;
}
.risk-low { background: rgba(57, 255, 20, 0.15); color: #39FF14; border: 1px solid #39FF14; }
.risk-med { background: rgba(255, 122, 0, 0.15); color: #FF7A00; border: 1px solid #FF7A00; }
.risk-high { background: rgba(255, 50, 50, 0.15); color: #FF3232; border: 1px solid #FF3232; box-shadow: 0 0 15px rgba(255, 50, 50, 0.3); }

div.stButton > button {
    background: #39FF14; color: #0A0A0A; border: none; border-radius: 8px;
    font-weight: 700; padding: .55rem 1.8rem;
    font-family: 'Orbitron', 'IBM Plex Sans Thai', sans-serif;
    box-shadow: 0 0 14px rgba(57,255,20,.35); transition: all .2s ease;
}
div.stButton > button:hover { background: #52FF33; box-shadow: 0 0 22px rgba(57,255,20,.55); }
</style>
""", unsafe_allow_html=True)

# ==================== Helpers ====================
def find_photo():
    try:
        files = os.listdir(".")
    except Exception:
        return None
    preferred = ["my_photo.jpg", "my_photo.png", "photo.jpg", "photo.png"]
    for name in preferred:
        if name in files:
            return name
    for f in sorted(files):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
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
    """สร้างและประเมินโมเดลทั้งหมด 12 วิธี"""
    df = make_data(20000)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    sm = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

    # 🎯 เพิ่มโมเดลเป็น 12 วิธี!
    models = {
        " Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "🌳 Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight="balanced"),
        " Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1),
        "👥 K-NN": KNeighborsClassifier(n_neighbors=5),
        "⚡ Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        "🎯 SVM (RBF)": SVC(probability=True, kernel='rbf', random_state=42, class_weight='balanced'),
        "🧠 Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True),
        "🎲 Naive Bayes": GaussianNB(),
        "🔺 AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "🌳 Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1),
        "🎰 Ensemble Voting": None,  # จะสร้างทีหลัง
        "🔍 Isolation Forest": IsolationForest(contamination=0.0017, random_state=42, n_jobs=-1)
    }
    
    # สร้าง Ensemble Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5))
        ],
        voting='soft'
    )
    models["🎰 Ensemble Voting"] = voting_clf

    trained, preds, probas, metrics = {}, {}, {}, {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, m) in enumerate(models.items()):
        status_text.text(f"⚙️ กำลังฝึก: {name}...")
        
        try:
            # Isolation Forest ใช้วิธีต่างกัน (unsupervised)
            if name == "🔍 Isolation Forest":
                m.fit(X_tr)  # ไม่ต้องใช้ SMOTE
                # แปลง prediction (-1 = anomaly, 1 = normal) ให้เป็น (1 = fraud, 0 = normal)
                pred_raw = m.predict(X_te)
                yp = np.where(pred_raw == -1, 1, 0)
                # คำนวณ probability จาก decision function
                pr = -m.decision_function(X_te)
                pr = (pr - pr.min()) / (pr.max() - pr.min())  # normalize to 0-1
            else:
                if name in ["👥 K-NN", "🎲 Naive Bayes"]:
                    m.fit(X_tr, y_tr)
                else:
                    m.fit(X_tr_res, y_tr_res)
                yp = m.predict(X_te)
                pr = m.predict_proba(X_te)[:, 1]
            
            trained[name] = m
            preds[name] = yp
            probas[name] = pr
            
            # คำนวณ metrics
            metrics[name] = {
                'Accuracy': accuracy_score(y_te, yp),
                'Precision': precision_score(y_te, yp, zero_division=0),
                'Recall': recall_score(y_te, yp, zero_division=0),
                'F1': f1_score(y_te, yp, zero_division=0),
                'AUC': roc_auc_score(y_te, pr)
            }
            
        except Exception as e:
            st.error(f" โมเดล {name} เกิดข้อผิดพลาด: {str(e)}")
            metrics[name] = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0, 'AUC': 0}
        
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.text("✅ เสร็จสิ้น!")
    
    # สร้าง DataFrame สำหรับแสดงผล
    comp = pd.DataFrame(metrics).T.reset_index()
    comp.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    best_name = comp.sort_values("F1", ascending=False).iloc[0]["Model"]
    
    return trained, scaler, comp, best_name, np.array(y_te), preds, probas, metrics

def make_comparison_plots(comp, y_te, preds, probas, best_name):
    """สร้างกราฟเปรียบเทียบ"""
    
    # 1. Bar Chart - F1 Score
    fig_f1, ax = plt.subplots(figsize=(12, 6))
    models_sorted = comp.sort_values("F1", ascending=True)
    colors = ['#39FF14' if m == best_name else '#3A465C' for m in models_sorted['Model']]
    bars = ax.barh(models_sorted['Model'], models_sorted['F1'], color=colors, alpha=0.8)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title(' F1 Score Comparison (โมเดลที่ดีที่สุด: ' + best_name + ')', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    for i, (idx, row) in enumerate(models_sorted.iterrows()):
        ax.text(row['F1'] + 0.02, i, f"{row['F1']:.3f}", va='center', fontsize=9)
    plt.tight_layout()
    
    # 2. ROC Curve - Top 6 Models
    fig_roc, ax = plt.subplots(figsize=(10, 8))
    top_models = comp.sort_values("AUC", ascending=False).head(6)
    for idx, row in top_models.iterrows():
        model_name = row['Model']
        fpr, tpr, _ = roc_curve(y_te, probas[model_name])
        auc_score = row['AUC']
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc_score:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(' ROC Curve - Top 6 Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    # 3. Confusion Matrix - Best Model
    cm = confusion_matrix(y_te, preds[best_name])
    fig_cm, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Greens', interpolation='nearest')
    ax.set_title(f' Confusion Matrix - {best_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Fraud'])
    ax.set_yticklabels(['Normal', 'Fraud'])
    
    # แสดงตัวเลขใน confusion matrix
    for i in range(2):
        for j in range(2):
            color = '#0A0A0A' if cm[i, j] > cm.max()/2 else '#D5DCEA'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', color=color, fontweight='bold', fontsize=12)
    
    fig_cm.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    
    # 4. Radar Chart - Top 3 Models
    fig_radar, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    top3 = comp.sort_values("F1", ascending=False).head(3)
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    colors_radar = ['#39FF14', '#6A3AB2', '#FF7A00']
    for idx, (_, row) in enumerate(top3.iterrows()):
        values = [row[cat] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('🎯 Performance Radar - Top 3 Models', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    
    return fig_f1, fig_roc, fig_cm, fig_radar

# ==================== Navigation ====================
NAV_OPTIONS = [" หน้าหลัก", "👤 ผู้พัฒนา"]

if "nav_widget" not in st.session_state:
    st.session_state.nav_widget = NAV_OPTIONS[0]

def go_home():
    st.session_state.nav_widget = NAV_OPTIONS[0]

def go_dev():
    st.session_state.nav_widget = NAV_OPTIONS[1]

st.sidebar.markdown('<div class="hub-title">MACHINE LEARNING HUB</div>', unsafe_allow_html=True)
st.sidebar.markdown('<hr class="hub-hr">', unsafe_allow_html=True)
st.sidebar.markdown("### 📍 นำทาง")
page = st.sidebar.selectbox("เลือกหน้า", NAV_OPTIONS, key="nav_widget")

tb1, tb2, _ = st.columns([1.2, 1.2, 6])
with tb1:
    st.button("🏠 หน้าหลัก", use_container_width=True, key="top_home_btn", on_click=go_home)
with tb2:
    st.button(" ผู้พัฒนา", use_container_width=True, key="top_dev_btn", on_click=go_dev)

# ================================================================
#              หน้าหลัก
# ================================================================
if page == "🏠 หน้าหลัก":
    st.markdown('<span class="tag-cyber">ADVANCED FRAUD DETECTION SYSTEM</span>', unsafe_allow_html=True)
    st.markdown('<div class="grad-title">ระบบตรวจจับธุรกรรม 12 โมเดล</div>', unsafe_allow_html=True)
    st.markdown('<div class="hazard-line"></div>', unsafe_allow_html=True)
    st.caption("🔍 การเปรียบเทียบประสิทธิภาพ 12 อัลกอริทึมสำหรับการตรวจจับธุรกรรมผิดปกติ")

    st.markdown("")
    
    # Initialize or load models
    if "eval" not in st.session_state:
        st.info("⏳ ยังไม่มีโมเดล — กดปุ่มด้านล่างเพื่อฝึกทั้ง 12 โมเดล")
        if st.button("🚀 เริ่มต้นฝึก 12 โมเดล", use_container_width=True, type="primary"):
            with st.spinner("⚙️ กำลังฝึก 12 โมเดล... อาจใช้เวลา 2-3 นาที"):
                st.session_state["eval"] = build_and_eval()
            st.rerun()
    else:
        trained, scaler, comp, best_name, y_te, preds, probas, metrics = st.session_state["eval"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(" ขนาดข้อมูล", "20,000 รายการ")
        m2.metric(" จำนวนโมเดล", "12 วิธี")
        m3.metric("🏆 โมเดลที่ดีที่สุด", best_name.split()[-2] + " " + best_name.split()[-1] if len(best_name.split()) > 1 else best_name)
        m4.metric("📈 F1 Score สูงสุด", f"{comp['F1'].max():.2%}")

        st.markdown("")

        t1, t2, t3, t4, t5 = st.tabs([
            "📚 รายชื่อโมเดล", "📊 ผลการประเมิน", "📈 กราฟเปรียบเทียบ", " ทดลองตรวจจับ", " คู่มือ"
        ])

        with t1:
            st.subheader(" 12 โมเดลที่ใช้ในการศึกษา")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**📊 Logistic Regression**")
                st.caption("แบบจำลองเชิงเส้น เหมาะเป็น baseline")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**🌳 Decision Tree**")
                st.caption("แบ่งข้อมูลแบบ tree structure ตีความง่าย")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**🌲 Random Forest**")
                st.caption("Ensemble ของ Decision Trees ลด overfitting")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**⚡ Gradient Boosting**")
                st.caption("Boosting algorithm ที่ทรงพลัง")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**🎯 SVM (RBF)**")
                st.caption("Support Vector Machine with RBF kernel")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**🧠 Neural Network**")
                st.caption("Multi-layer Perceptron (Deep Learning)")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**👥 K-NN**")
                st.caption("K-Nearest Neighbors - distance-based")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**🎲 Naive Bayes**")
                st.caption("Probabilistic classifier แบบง่ายและเร็ว")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**🔺 AdaBoost**")
                st.caption("Adaptive Boosting - ปรับน้ำหนักข้อมูลผิด")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**🌳 Extra Trees**")
                st.caption("Extremely Randomized Trees - สุ่มมากกว่า RF")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**🎰 Ensemble Voting**")
                st.caption("รวมผลจาก LR + RF + Gradient Boosting")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("**🔍 Isolation Forest**")
                st.caption("Anomaly Detection - unsupervised learning")
                st.markdown('</div>', unsafe_allow_html=True)

        with t2:
            st.subheader("📊 ตารางเปรียบเทียบประสิทธิภาพ")
            st.dataframe(comp.round(4).sort_values("F1", ascending=False), 
                        use_container_width=True, hide_index=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"🏆 **โมเดลที่ดีที่สุด (F1-Score):** {best_name}")
            with c2:
                st.info(f" **AUC สูงสุด:** {comp['AUC'].max():.4f}")

        with t3:
            st.subheader(" กราฟเปรียบเทียบโมเดล")
            fig_f1, fig_roc, fig_cm, fig_radar = make_comparison_plots(comp, y_te, preds, probas, best_name)
            
            st.pyplot(fig_f1)
            
            g1, g2 = st.columns(2)
            with g1:
                st.pyplot(fig_roc)
            with g2:
                st.pyplot(fig_cm)
            
            st.pyplot(fig_radar)

        with t5:
            st.subheader("📖 คู่มือการใช้ระบบ")
            st.markdown("""
            ### 🎯 วิธีการใช้งาน
            
            1. **ฝึกโมเดล**: กดปุ่ม "เริ่มต้นฝึก 12 โมเดล" ในครั้งแรก
            2. **ดูผลการประเมิน**: เลือกแท็บ "ผลการประเมิน" และ "กราฟเปรียบเทียบ"
            3. **ทดลองตรวจจับ**: ไปที่แท็บ "ทดลองตรวจจับ"
            
            ### 📝 การกรอกข้อมูล
            
            - **Time**: เวลาที่เกิดธุรกรรม (วินาที)
            - **Amount**: จำนวนเงิน (USD)
            - **V1, V2**: คุณลักษณะจาก PCA (ค่ามาตรฐาน)
            
            ### 🎲 ตัวอย่างข้อมูล
            
            - **Normal**: ธุรกรรมปกติ ทั่วไป
            - **Fraud**: ธุรกรรมผิดปกติ มักมี Amount สูง, V1/V2 ผิดปกติ
            - **สุ่ม**: สร้างข้อมูลแบบสุ่ม
            
            ### 📊 การแปลผล
            
            - **✅ Normal**: ความน่าจะเป็น Fraud < 50%
            - **🚨 Fraud**: ความน่าจะเป็น Fraud ≥ 50%
            - **Risk Level**: แสดงระดับความเสี่ยง (ต่ำ/กลาง/สูง)
            """)

        # ================================================================
        # 🔮 แท็บทดลองตรวจจับ
        # ================================================================
        with t4:
            st.subheader("🔮 ทดลองตรวจจับธุรกรรม")
            
            # Initialize session state
            if "tx_history" not in st.session_state:
                st.session_state.tx_history = []
            if "inp_time" not in st.session_state: 
                st.session_state.inp_time = 50000.0
            if "inp_amount" not in st.session_state: 
                st.session_state.inp_amount = 100.0
            if "inp_v1" not in st.session_state: 
                st.session_state.inp_v1 = 0.0
            if "inp_v2" not in st.session_state: 
                st.session_state.inp_v2 = 0.0

            model_name = st.selectbox(
                "🎛️ เลือกโมเดล", 
                list(trained.keys()),
                index=list(trained.keys()).index(best_name) if best_name in trained.keys() else 0
            )

            st.markdown("**📝 กรอกข้อมูลธุรกรรม**")
            
            # 🎲 Preset Examples
            st.markdown("<small style='color:#93A1B8'>⚡ โหลดข้อมูลตัวอย่างด่วน:</small>", unsafe_allow_html=True)
            pc1, pc2, pc3, pc4 = st.columns(4)
            
            with pc1:
                if st.button("✅ Normal", use_container_width=True):
                    st.session_state.inp_time = float(np.random.uniform(10000, 150000))
                    st.session_state.inp_amount = float(np.random.exponential(50))
                    st.session_state.inp_v1 = float(np.random.normal(0, 1))
                    st.session_state.inp_v2 = float(np.random.normal(0, 1))
                    st.rerun()
            
            with pc2:
                if st.button(" Fraud", use_container_width=True):
                    st.session_state.inp_time = float(np.random.uniform(0, 5000))
                    st.session_state.inp_amount = float(np.random.exponential(500) + 500)
                    st.session_state.inp_v1 = float(np.random.normal(-3, 1.5))
                    st.session_state.inp_v2 = float(np.random.normal(2.5, 1.5))
                    st.rerun()
            
            with pc3:
                if st.button("🎲 สุ่ม", use_container_width=True):
                    st.session_state.inp_time = float(np.random.uniform(0, 172792))
                    st.session_state.inp_amount = float(np.random.exponential(88))
                    st.session_state.inp_v1 = float(np.random.randn())
                    st.session_state.inp_v2 = float(np.random.randn())
                    st.rerun()
            
            with pc4:
                if st.button("🔍 ทดสอบทุกโมเดล", use_container_width=True):
                    st.session_state.run_all_models = True
                else:
                    st.session_state.run_all_models = False

            c1, c2 = st.columns(2)
            with c1:
                time_val = st.number_input("⏱️ Time (วินาที)", 0.0, 200000.0, 
                                          value=st.session_state.inp_time, key="ni_time")
                amount = st.number_input("💰 Amount (USD)", 0.0, 50000.0, 
                                        value=st.session_state.inp_amount, key="ni_amount")
            with c2:
                v1 = st.number_input(" V1 (PCA)", -50.0, 50.0, 
                                    value=st.session_state.inp_v1, key="ni_v1")
                v2 = st.number_input(" V2 (PCA)", -50.0, 50.0, 
                                    value=st.session_state.inp_v2, key="ni_v2")

            if st.button("🔍 ตรวจจับธุรกรรม", use_container_width=True, type="primary"):
                # Update session state
                st.session_state.inp_time = float(time_val)
                st.session_state.inp_amount = float(amount)
                st.session_state.inp_v1 = float(v1)
                st.session_state.inp_v2 = float(v2)

                # Prepare input
                inp_dict = {f"V{i}": 0.0 for i in range(1, 29)}
                inp_dict["V1"] = float(v1)
                inp_dict["V2"] = float(v2)
                scaled = scaler.transform(pd.DataFrame([[amount, time_val]], 
                                                      columns=["Amount", "Time"]))[0]
                inp_dict["Amount"] = float(scaled[0])
                inp_dict["Time"] = float(scaled[1])
                train_columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
                inp = pd.DataFrame([inp_dict])[train_columns]

                # 🎯 ทดสอบทุกโมเดลหรือแค่โมเดลที่เลือก
                if st.session_state.get("run_all_models", False):
                    st.markdown("###  ผลการตรวจจับจากทุกโมเดล")
                    
                    results = {}
                    for m_name, m_model in trained.items():
                        try:
                            pred = int(m_model.predict(inp)[0])
                            proba = float(m_model.predict_proba(inp)[0][1]) if hasattr(m_model, 'predict_proba') else 0.5
                            results[m_name] = {'pred': pred, 'proba': proba}
                        except:
                            results[m_name] = {'pred': -1, 'proba': 0}
                    
                    # แสดงผล
                    cols = st.columns(3)
                    for idx, (m_name, res) in enumerate(results.items()):
                        with cols[idx % 3]:
                            if res['pred'] == 1:
                                st.markdown(f'<div style="background:rgba(255,50,50,0.1);border:1px solid #FF3232;border-radius:8px;padding:10px;margin:5px 0;">', 
                                          unsafe_allow_html=True)
                                st.markdown(f"**{m_name.split()[-1]}**")
                                st.markdown('<span style="color:#FF3232;font-weight:bold;">🚨 FRAUD</span>', 
                                          unsafe_allow_html=True)
                                st.caption(f"Prob: {res['proba']:.1%}")
                            else:
                                st.markdown(f'<div style="background:rgba(57,255,20,0.1);border:1px solid #39FF14;border-radius:8px;padding:10px;margin:5px 0;">', 
                                          unsafe_allow_html=True)
                                st.markdown(f"**{m_name.split()[-1]}**")
                                st.markdown('<span style="color:#39FF14;font-weight:bold;">✅ NORMAL</span>', 
                                          unsafe_allow_html=True)
                                st.caption(f"Prob: {res['proba']:.1%}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # สรุปผล
                    fraud_count = sum(1 for r in results.values() if r['pred'] == 1)
                    st.markdown("### 📊 สรุปผล")
                    if fraud_count > len(results) / 2:
                        st.error(f"🚨 **เสียงข้างมากตรวจจับเป็น Fraud:** {fraud_count}/{len(results)} โมเดล")
                    else:
                        st.success(f"✅ **เสียงข้างมากตรวจจับเป็น Normal:** {len(results)-fraud_count}/{len(results)} โมเดล")
                
                else:
                    # ทดสอบโมเดลเดียว
                    m = trained[model_name]
                    pred = int(m.predict(inp)[0])
                    proba = float(m.predict_proba(inp)[0][1]) if hasattr(m, 'predict_proba') else 0.5

                    # AI Explanation
                    reasons = []
                    if amount > 300:
                        reasons.append("💰 จำนวนเงินสูงกว่าค่าเฉลี่ย (>300 USD)")
                    if abs(v1) > 2.5:
                        reasons.append(f"📉 ค่า V1 ผิดปกติ ({v1:.2f})")
                    if abs(v2) > 2.5:
                        reasons.append(f"📈 ค่า V2 ผิดปกติ ({v2:.2f})")
                    if time_val < 10000:
                        reasons.append("⏱️ เกิดในช่วงเวลาเสี่ยง")
                    if not reasons:
                        reasons.append("✅ ไม่พบความผิดปกติชัดเจน")

                    # Add to history
                    new_record = {
                        "เวลา": datetime.datetime.now().strftime("%H:%M:%S"),
                        "โมเดล": model_name.split()[-1],
                        "Amount": f"${amount:,.2f}",
                        "ผล": "🚨 FRAUD" if pred == 1 else "✅ NORMAL",
                        "ความน่าจะเป็น": f"{proba:.1%}"
                    }
                    st.session_state.tx_history.insert(0, new_record)
                    if len(st.session_state.tx_history) > 10:
                        st.session_state.tx_history.pop()

                    st.markdown("---")
                    
                    # แสดงผล
                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        if pred == 1:
                            risk_level = "HIGH" if proba > 0.8 else "MEDIUM"
                            badge_class = "risk-high"
                            st.markdown(f'<div class="risk-badge {badge_class}">🚨 FRAUD</div>', 
                                      unsafe_allow_html=True)
                        else:
                            risk_level = "LOW"
                            badge_class = "risk-low"
                            st.markdown(f'<div class="risk-badge {badge_class}">✅ NORMAL</div>', 
                                      unsafe_allow_html=True)
                        
                        st.markdown(f"<p style='text-align:center;margin:15px 0 5px 0;color:#93A1B8;'>ความน่าจะเป็น</p>", 
                                  unsafe_allow_html=True)
                        bar_color = "#FF3232" if pred == 1 else "#39FF14"
                        st.markdown(f"<h2 style='text-align:center;color:{bar_color};margin:0;'>{proba:.1%}</h2>", 
                                  unsafe_allow_html=True)
                        
                        # Custom progress bar
                        st.markdown(f"""
                        <div style="background:#1A2230;border-radius:10px;height:15px;width:100%;overflow:hidden;border:1px solid #3A465C;">
                            <div style="background:linear-gradient(90deg,{bar_color},#6A3AB2);height:100%;width:{proba*100}%;border-radius:10px;box-shadow:0 0 10px {bar_color};transition:width 1s;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"🎯 Risk Level: **{risk_level}**")

                    with res_col2:
                        st.markdown("**🧠 AI Analysis**")
                        for reason in reasons:
                            st.markdown(f"- {reason}")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.info(f"📊 โมเดล: **{model_name}**")

            # History Log
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📜 ประวัติการทดสอบ", expanded=True):
                if len(st.session_state.tx_history) > 0:
                    hist_df = pd.DataFrame(st.session_state.tx_history)
                    st.dataframe(hist_df, use_container_width=True, hide_index=True)
                    if st.button("🗑️ ล้างประวัติ"):
                        st.session_state.tx_history = []
                        st.rerun()
                else:
                    st.caption("ยังไม่มีประวัติ")

    st.markdown("---")
    st.caption("📚 Machine Learning Project • 🛠️ Python, scikit-learn, Streamlit")

# ================================================================
#  หน้าผู้พัฒนา
# ================================================================
else:
    st.markdown("## 👤 ผู้พัฒนา")
    if PHOTO:
        st.image(PHOTO, width=200)
    st.markdown("""
    **นาย จตุรภัทร สถาปีตานนท์**  
    รหัสนักศึกษา: 664245024  
    สาขา: วิทยาการคอมพิวเตอร์
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ เทคโนโลยี")
    st.markdown("""
    - 🐍 Python
    - 🤖 scikit-learn
    - 🚀 Streamlit
    - 📊 matplotlib
    -  pandas
    - ⚖️ imbalanced-learn
    """)