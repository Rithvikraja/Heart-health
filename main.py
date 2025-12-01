import streamlit as st
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -------------------------------------------
# PAGE CONFIG + GLOBAL UI STYLING
# -------------------------------------------
st.set_page_config(page_title="Heart Health Studio", page_icon="❤️", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
body {background: #f5f7fa;}
.block-card {background: black; padding: 25px; border-radius: 14px; margin-bottom: 15px;}
div.stButton > button {
    background: linear-gradient(135deg, #e53935, #c62828) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 22px;
    font-size: 18px !important;
    font-weight: 600 !important;
    border: none;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #b71c1c, #8e0000) !important;
}
.css-1aumxhk, .stSlider, .stSelectbox {border-radius: 10px !important;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# HEADER
# -------------------------------------------
st.markdown("""
<div style='text-align:center; margin-bottom: 20px;'>
    <div style='font-size:40px; font-weight:700; color:Gold;'>❤️ Heart Health Studio</div>
    <div style='font-size:17px; color:#4b5563; margin-top:4px;'>A compact dashboard that turns your health numbers into quiet insights.</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
@st.cache_data
def load():
    return pd.read_csv("heart.csv")

df = load()

# -------------------------------------------
# SECTION 1 — HEALTH PATTERNS
# -------------------------------------------
st.markdown("<div class='block-card'>", unsafe_allow_html=True)
st.subheader("📊 Health Patterns")
c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["age"], bins=10, color="#4F46E5", alpha=0.85, edgecolor="black", linewidth=1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Age Spread", fontsize=14, color="#1E1B4B")
    ax.set_xlabel("Age Range", fontsize=11)
    ax.set_ylabel("Number of Individuals", fontsize=11)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("white")
    st.pyplot(fig)

with c2:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(df["cholesterol"], df["heart_rate"], alpha=0.7)
    ax2.set_xlabel("Cholesterol")
    ax2.set_ylabel("Heart Rate")
    ax2.set_title("Cholesterol vs Heart Rate")
    st.pyplot(fig2)

st.markdown("</div>", unsafe_allow_html=True)


# ❤️ HEART INFORMATION SECTION (FULLY WORKING)

st.subheader("❤️ Heart Information")

with st.expander("Click to expand — Learn about the heart, risks & wellness"):
    st.markdown("""
    ### 🫀 What the Heart Does
    The heart works around the clock — pushing blood, oxygen, and nutrients throughout your body.
    It adjusts its rhythm quietly depending on your activity, stress, sleep, and emotions.

    ### ⚠ Common Risk Factors
    - High blood pressure  
    - High cholesterol  
    - Diabetes  
    - Smoking  
    - Stress  
    - Obesity / High BMI  
    - Sedentary lifestyle  
    - Family history  

    ### 🌿 Tips for a Healthier Heart
    - Add 20–30 mins of walking or mild exercise daily  
    - Eat more vegetables, fruits, and whole grains  
    - Reduce salty and oily foods  
    - Try deep breathing for stress  
    - Maintain a regular sleep cycle  
    - Avoid smoking and excess alcohol  
    - Schedule routine health checkups  

    ❤️ *Small habits compound into stronger heart health over time.*  
    """)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------
# SECTION 2 — TRAIN MODEL
# -------------------------------------------
st.subheader("Train Your Model")
features = df.drop("target", axis=1)
labels = df["target"]

if st.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    st.session_state["model"] = model
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model trained! Accuracy: {acc*100:.2f}%")



# -------------------------------------------
# SECTION 3 — PREDICTOR
# -------------------------------------------
st.markdown("<div class='block-card'>", unsafe_allow_html=True)
st.subheader("🔮 Predict Heart Risk")

name = st.text_input("Name")
age = st.slider("Age", 0, 90)
gender = st.selectbox("Gender", ["Male", "Female"])
rest = st.slider("Resting BP", 60, 200)
hr = st.slider("Heart Rate", 40, 200)
chol = st.slider("Cholesterol", 120, 350)
stress = st.slider("Stress Level", 1, 5)
smoker = st.selectbox("Smoker?", ["No", "Yes"])
diab = st.selectbox("Diabetes?", ["No", "Yes"])
bmi = st.slider("BMI", 15.0, 40.0)

gender_val = 1 if gender == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0
diab_val = 1 if diab == "Yes" else 0
inp = [[age, gender_val, rest, hr, chol, stress, smoker_val, diab_val, bmi]]

# Centered predict button
cA, cB, cC = st.columns([1, 2, 1])
with cB:
    predict_pressed = st.button("Predict", use_container_width=True)

if predict_pressed:
    if "model" not in st.session_state:
        st.error("Train the model first!")
    else:
        model = st.session_state["model"]
        res = model.predict(inp)[0]

        if res == 1:
            st.error(f"⚠High Risk Detected")
            st.warning("### Recommended Tips\n"
                       "- Walk 20–30 minutes daily.\n"
                       "- Eat more fruits, vegetables & whole grains.\n"
                       "- Reduce salty and fried foods.\n"
                       "- If you smoke, consider reducing/quitting.\n"
                       "- Practice breathing exercises.\n"
                       "- Maintain consistent sleep.\n"
                       "- Get regular health checkups.\n")
        else:
            st.success(f"✅Low Risk")

        # PDF report
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(50, 750, "Heart Health Report")
        c.setFont("Helvetica", 14)
        c.drawString(50, 720, f"Name: {name}")
        c.drawString(50, 700, f"Age: {age}")
        c.drawString(50, 680, f"Gender: {gender}")
        c.drawString(50, 660, f"Rest BP: {rest}")
        c.drawString(50, 640, f"Heart Rate: {hr}")
        c.drawString(50, 620, f"Cholesterol: {chol}")
        c.drawString(50, 600, f"Stress: {stress}")
        c.drawString(50, 580, f"Smoker: {smoker}")
        c.drawString(50, 560, f"Diabetes: {diab}")
        c.drawString(50, 540, f"BMI: {bmi}")
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, 500, f"Prediction: {'High Risk' if res == 1 else 'Low Risk'}")
        if res == 1:
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, 470, "Recommended Tips:")
            c.setFont("Helvetica", 13)
            tips = [
                "• Daily 20–30 min walking.",
                "• Eat more fruits & vegetables.",
                "• Reduce salty & oily foods.",
                "• Consider quitting smoking.",
                "• Practice slow breathing.",
                "• Maintain sleep schedule.",
                "• Get regular checkups.",
            ]
            y = 450
            for t in tips:
                c.drawString(60, y, t)
                y -= 20
        c.save()
        pdf_buffer.seek(0)
        st.download_button(label="⬇ Download Report (PDF)", data=pdf_buffer, file_name="heart_report.pdf", mime="application/pdf", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------
# SECTION 4 — BMI CALCULATOR + METER
# -------------------------------------------
st.markdown("<div class='block-card'>", unsafe_allow_html=True)
left_col, right_col = st.columns([1.2, 1])

# BMI CALCULATOR
with left_col:
    st.markdown("### Quick BMI Calculator")
    w = st.number_input("Weight (kg)", 1.0, 200.0, step=0.5, key="weight")
    h = st.number_input("Height (cm)", 50.0, 250.0, step=1.0, key="height")

    if st.button("Calculate BMI", key="calc_bmi"):
        height_m = h / 100
        calculated_bmi = w / (height_m ** 2)
        st.session_state["bmi_value"] = calculated_bmi
        st.success(f"Your BMI: **{calculated_bmi:.2f}**")

# BMI METER
with right_col:
    st.markdown("### BMI Meter")
    if "bmi_value" in st.session_state:
        bmi_val = st.session_state["bmi_value"]
        pct = int(((bmi_val - 15) / (40 - 15)) * 100)
        pct = max(0, min(100, pct))
        if bmi_val < 18.5:
            color = "blue"; status = "Underweight"
        elif bmi_val < 24.9:
            color = "green"; status = "Normal"
        elif bmi_val < 29.9:
            color = "orange"; status = "Overweight"
        else:
            color = "red"; status = "Obese"
        st.markdown(f"""
        <div style="
            width: 100%;
            height: 28px;
            background: #e5e7eb;
            border-radius: 14px;
            overflow: hidden;
            margin-bottom: 8px;">
            <div style="
                width:{pct}%;
                height:100%;
                background:{color};
                border-radius: 14px;
                transition: width 0.6s;">
            </div>
        </div>
        <div style='text-align:center; font-size:20px; font-weight:600; color:White;'>{status}</div>
        <div style='text-align:center; font-size:18px; font-weight:500; color:White;'>BMI: {bmi_val:.2f}</div>
        """, unsafe_allow_html=True)
    else:
        st.info("Calculate BMI to see your meter.")

st.markdown("</div>", unsafe_allow_html=True)
# -------------------------------------------
# SECTION — CHOLESTEROL CALCULATOR
# -------------------------------------------
st.markdown("<div class='block-card'>", unsafe_allow_html=True)
st.subheader("🩸 Cholesterol Calculator")

colA, colB = st.columns(2)

with colA:
    hdl = st.number_input("HDL (Good Cholesterol)", 10, 120, step=1)
    ldl = st.number_input("LDL (Bad Cholesterol)", 20, 250, step=1)
    trig = st.number_input("Triglycerides", 30, 500, step=1)

with colB:
    st.markdown("### Formula")
    st.markdown("""
    **Total Cholesterol = HDL + LDL + (Triglycerides / 5)**  
    """)

if st.button("Calculate Total Cholesterol"):
    total_chol = hdl + ldl + (trig / 5)
    st.success(f"**Your Total Cholesterol: {total_chol:.1f} mg/dL**")

    # Category Classification
    if total_chol < 200:
        level = "Desirable"
        color = "green"
        msg = "Healthy range. Maintain your lifestyle!"
    elif 200 <= total_chol <= 239:
        level = "Borderline High"
        color = "orange"
        msg = "A bit elevated. Watch your food choices."
    else:
        level = "High"
        color = "red"
        msg = "Risky level. Consider consulting a doctor."

    st.markdown(f"""
        <div style="
            background:{color};
            padding:14px;
            border-radius:10px;
            text-align:center;
            font-size:20px;
            font-weight:600;
            color:white;
            margin-top:10px;">
            {level}
        </div>
        <div style="margin-top:8px; font-size:16px; color:#e5e7eb;">
            {msg}
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


