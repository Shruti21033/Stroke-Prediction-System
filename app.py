# ============================================================
# 🩺 INTELLIGENT STROKE RISK PREDICTION SYSTEM
# Complete Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f8fc;
    }

    /* Main container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Main title */
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #173b57;
        text-align: center;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .main-subtitle {
        font-size: 17px;
        color: #64748b;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Cards */
    .custom-card {
        background-color: white;
        padding: 22px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }

    /* Risk cards */
    .low-risk {
        background-color: #dcfce7;
        border-left: 6px solid #22c55e;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }

    .medium-risk {
        background-color: #fef3c7;
        border-left: 6px solid #f59e0b;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }

    .high-risk {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }

    .risk-title {
        font-size: 30px;
        font-weight: 800;
    }

    .risk-description {
        font-size: 15px;
        margin-top: 8px;
    }

    /* Section heading */
    .section-title {
        color: #173b57;
        font-size: 25px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Small information cards */
    .info-box {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
        text-align: center;
    }

    .info-number {
        font-size: 28px;
        font-weight: bold;
        color: #173b57;
    }

    .info-label {
        font-size: 14px;
        color: #64748b;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding-top: 30px;
        padding-bottom: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = "stroke_pipeline.pkl"

    if not os.path.exists(model_path):
        return None

    return joblib.load(model_path)


try:

    model = load_model()

    if model is None:

        st.error("❌ stroke_pipeline.pkl was not found.")

        st.info(
            "Please place stroke_pipeline.pkl in the same folder "
            "as app.py."
        )

        st.stop()

except Exception as e:

    st.error("❌ Unable to load the machine learning model.")

    st.error(f"Model error: {e}")

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


if "last_prediction" not in st.session_state:

    st.session_state.last_prediction = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🩺 Intelligent Stroke Risk Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-Powered Supervised Learning Based Stroke Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3004/3004458.png",
        width=80
    )

    st.title("🩺 Stroke AI")

    st.caption("Intelligent Healthcare Decision Support")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "👤 Risk Assessment",
            "📊 Analytics",
            "📁 Batch Prediction",
            "ℹ️ About System"
        ]
    )

    st.divider()

    st.markdown("### 🤖 Machine Learning")

    st.write("• Logistic Regression")

    st.write("• Random Forest")

    st.divider()

    st.warning(
        "This system is for educational and "
        "decision-support purposes only."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def calculate_age_group(age):

    if age < 30:

        return "Young"

    elif age < 60:

        return "Middle Age"

    else:

        return "Senior"


def calculate_bmi_category(bmi):

    if bmi < 18.5:

        return "Underweight"

    elif bmi < 25:

        return "Normal"

    elif bmi < 30:

        return "Overweight"

    else:

        return "Obese"


def calculate_risk_factor_count(
    hypertension,
    heart_disease,
    age,
    avg_glucose_level
):

    return (
        int(hypertension)
        + int(heart_disease)
        + int(age >= 60)
        + int(avg_glucose_level >= 140)
    )


def get_risk_level(probability):

    if probability < 30:

        return "LOW"

    elif probability < 70:

        return "MEDIUM"

    else:

        return "HIGH"


def get_risk_description(risk_level):

    if risk_level == "LOW":

        return (
            "The model estimates a relatively lower "
            "stroke probability based on the provided information."
        )

    elif risk_level == "MEDIUM":

        return (
            "The model estimates a moderate stroke probability. "
            "The highlighted risk factors should be reviewed."
        )

    else:

        return (
            "The model estimates a higher stroke probability. "
            "The highlighted patient factors require attention."
        )


def get_risk_class(risk_level):

    if risk_level == "LOW":

        return "low-risk"

    elif risk_level == "MEDIUM":

        return "medium-risk"

    else:

        return "high-risk"


def create_patient_dataframe(
    gender,
    age,
    hypertension,
    heart_disease,
    ever_married,
    work_type,
    residence_type,
    avg_glucose_level,
    bmi,
    smoking_status
):

    age_group = calculate_age_group(age)

    bmi_category = calculate_bmi_category(bmi)

    risk_factor_count = calculate_risk_factor_count(
        hypertension,
        heart_disease,
        age,
        avg_glucose_level
    )

    patient = pd.DataFrame({

        "gender": [gender],

        "age": [age],

        "hypertension": [hypertension],

        "heart_disease": [heart_disease],

        "ever_married": [ever_married],

        "work_type": [work_type],

        "Residence_type": [residence_type],

        "avg_glucose_level": [avg_glucose_level],

        "bmi": [bmi],

        "smoking_status": [smoking_status],

        "age_group": [age_group],

        "bmi_category": [bmi_category],

        "risk_factor_count": [risk_factor_count]

    })

    return patient


def make_prediction(patient):

    prediction = model.predict(patient)[0]

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(patient)[0][1] * 100

    else:

        probability = float(prediction) * 100

    risk_level = get_risk_level(probability)

    return prediction, probability, risk_level


def create_risk_gauge(probability):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={
                "suffix": "%",
                "font": {
                    "size": 38
                }
            },
            title={
                "text": "Stroke Probability"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "thickness": 0.25
                },
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "#dcfce7"
                    },
                    {
                        "range": [30, 70],
                        "color": "#fef3c7"
                    },
                    {
                        "range": [70, 100],
                        "color": "#fee2e2"
                    }
                ],
                "threshold": {
                    "line": {
                        "width": 5
                    },
                    "thickness": 0.8,
                    "value": probability
                }
            }
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


def create_risk_factor_chart(
    age,
    bmi,
    avg_glucose_level,
    hypertension,
    heart_disease
):

    factors = [
        "Age",
        "BMI",
        "Glucose",
        "Hypertension",
        "Heart Disease"
    ]

    values = [
        min(age, 100),
        min(bmi * 2, 100),
        min(avg_glucose_level / 2, 100),
        hypertension * 100,
        heart_disease * 100
    ]

    df = pd.DataFrame({
        "Factor": factors,
        "Value": values
    })

    fig = px.bar(
        df,
        x="Factor",
        y="Value",
        title="Patient Risk Factor Overview"
    )

    fig.update_layout(
        height=350,
        yaxis_title="Relative Indicator",
        xaxis_title=""
    )

    return fig


def generate_explanation(
    age,
    bmi,
    avg_glucose_level,
    hypertension,
    heart_disease,
    smoking_status
):

    explanations = []

    if age >= 60:

        explanations.append(
            "🔴 Age is 60 years or above."
        )

    elif age >= 45:

        explanations.append(
            "🟡 Age is within a middle-to-older age range."
        )

    else:

        explanations.append(
            "🟢 Age is below 45 years."
        )


    if hypertension == 1:

        explanations.append(
            "🔴 Hypertension is marked as present."
        )

    else:

        explanations.append(
            "🟢 Hypertension is not marked as present."
        )


    if heart_disease == 1:

        explanations.append(
            "🔴 Heart disease is marked as present."
        )

    else:

        explanations.append(
            "🟢 Heart disease is not marked as present."
        )


    if avg_glucose_level >= 140:

        explanations.append(
            "🔴 Average glucose level is 140 or above."
        )

    elif avg_glucose_level >= 100:

        explanations.append(
            "🟡 Average glucose level is between 100 and 139."
        )

    else:

        explanations.append(
            "🟢 Average glucose level is below 100."
        )


    if bmi >= 30:

        explanations.append(
            "🔴 BMI falls in the obese category."
        )

    elif bmi >= 25:

        explanations.append(
            "🟡 BMI falls in the overweight category."
        )

    elif bmi < 18.5:

        explanations.append(
            "🟡 BMI falls in the underweight category."
        )

    else:

        explanations.append(
            "🟢 BMI falls in the normal category."
        )


    if smoking_status == "smokes":

        explanations.append(
            "🔴 Current smoking status is reported."
        )

    elif smoking_status == "formerly smoked":

        explanations.append(
            "🟡 Former smoking history is reported."
        )

    else:

        explanations.append(
            "🟢 No current smoking status is reported."
        )


    return explanations


def save_prediction_history(
    age,
    bmi,
    glucose,
    probability,
    risk_level,
    prediction
):

    record = {

        "Date & Time":
        datetime.now().strftime("%d-%m-%Y %H:%M"),

        "Age":
        age,

        "BMI":
        bmi,

        "Glucose":
        glucose,

        "Probability (%)":
        round(probability, 2),

        "Risk Level":
        risk_level,

        "Prediction":
        (
            "Higher Risk"
            if prediction == 1
            else "Lower Risk"
        )
    }

    st.session_state.prediction_history.append(record)


# ============================================================
# PAGE 1 - DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">🏠 System Dashboard</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.prediction_history


    # --------------------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------------------

    total_predictions = len(history)

    if total_predictions > 0:

        high_count = sum(
            1
            for x in history
            if x["Risk Level"] == "HIGH"
        )

        medium_count = sum(
            1
            for x in history
            if x["Risk Level"] == "MEDIUM"
        )

        low_count = sum(
            1
            for x in history
            if x["Risk Level"] == "LOW"
        )

    else:

        high_count = 0

        medium_count = 0

        low_count = 0


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            f"""
            <div class="info-box">
                <div class="info-number">
                    {total_predictions}
                </div>
                <div class="info-label">
                    Total Assessments
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="info-box">
                <div class="info-number">
                    {high_count}
                </div>
                <div class="info-label">
                    Higher Risk
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="info-box">
                <div class="info-number">
                    {medium_count}
                </div>
                <div class="info-label">
                    Medium Risk
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="info-box">
                <div class="info-number">
                    {low_count}
                </div>
                <div class="info-label">
                    Low Risk
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="custom-card">

        ## 🧠 About the System

        The application takes relevant user information as input, 
        processes it using a trained machine learning model, 
        and provides a stroke prediction along with an estimated probability 
        and risk level.The system helps demonstrate how machine learning can be used to 
        analyze health-related data and identify potential stroke risk patterns.

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.markdown("## 🚀 System Features")

    f1, f2, f3 = st.columns(3)


    with f1:

        st.info(
            """
            ### 👤 Patient Assessment

            Enter patient information and
            generate an individual stroke
            risk assessment.
            """
        )


    with f2:

        st.info(
            """
            ### 📊 Risk Analytics

            Visualize prediction probability,
            risk factors and assessment history.
            """
        )


    with f3:

        st.info(
            """
            ### 📁 Batch Prediction

            Upload multiple patient records
            and generate predictions in bulk.
            """
        )


    # --------------------------------------------------------
    # QUICK START
    # --------------------------------------------------------

    st.markdown("## 🏃 Quick Start")

    st.write(
        """
        **Step 1:** Open **Risk Assessment**

        **Step 2:** Enter patient information

        **Step 3:** Click **Assess Stroke Risk**

        **Step 4:** Review probability and risk factors

        **Step 5:** Open **Analytics** to review prediction history
        """
    )


# ============================================================
# PAGE 2 - RISK ASSESSMENT
# ============================================================

elif page == "👤 Risk Assessment":

    st.markdown(
        '<div class="section-title">👤 Patient Risk Assessment</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter patient information to estimate stroke risk."
    )


    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------

    st.markdown("### 🧾 Patient Information")


    col1, col2, col3 = st.columns(3)


    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )


        age = st.slider(
            "Age",
            min_value=1,
            max_value=100,
            value=45,
            step=1
        )


        bmi = st.slider(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )


    with col2:

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )


        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )


        ever_married = st.selectbox(
            "Ever Married",
            [
                "Yes",
                "No"
            ]
        )


    with col3:

        avg_glucose_level = st.slider(
            "Average Glucose Level",
            min_value=50.0,
            max_value=300.0,
            value=100.0,
            step=0.1
        )


        smoking_status = st.selectbox(
            "Smoking Status",
            [
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown"
            ]
        )


        residence_type = st.selectbox(
            "Residence Type",
            [
                "Urban",
                "Rural"
            ]
        )


    # --------------------------------------------------------
    # WORK TYPE
    # --------------------------------------------------------

    work_type = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked"
        ]
    )


    st.divider()


    # --------------------------------------------------------
    # LIVE PATIENT SUMMARY
    # --------------------------------------------------------

    st.markdown("### 👀 Live Patient Summary")


    age_group_preview = calculate_age_group(age)

    bmi_category_preview = calculate_bmi_category(bmi)

    risk_factor_preview = calculate_risk_factor_count(
        hypertension,
        heart_disease,
        age,
        avg_glucose_level
    )


    s1, s2, s3, s4 = st.columns(4)


    with s1:

        st.metric(
            "Age Group",
            age_group_preview
        )


    with s2:

        st.metric(
            "BMI Category",
            bmi_category_preview
        )


    with s3:

        st.metric(
            "Risk Factors",
            risk_factor_preview
        )


    with s4:

        if avg_glucose_level >= 140:

            glucose_status = "Elevated"

        elif avg_glucose_level >= 100:

            glucose_status = "Moderate"

        else:

            glucose_status = "Normal"

        st.metric(
            "Glucose Status",
            glucose_status
        )


    st.write("")


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 ASSESS STROKE RISK",
        type="primary",
        use_container_width=True
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        try:

            patient = create_patient_dataframe(
                gender,
                age,
                hypertension,
                heart_disease,
                ever_married,
                work_type,
                residence_type,
                avg_glucose_level,
                bmi,
                smoking_status
            )


            prediction, probability, risk_level = make_prediction(
                patient
            )


            age_group = calculate_age_group(age)

            bmi_category = calculate_bmi_category(bmi)

            risk_factor_count = calculate_risk_factor_count(
                hypertension,
                heart_disease,
                age,
                avg_glucose_level
            )


            # Save latest prediction

            st.session_state.last_prediction = {

                "prediction":
                prediction,

                "probability":
                probability,

                "risk_level":
                risk_level,

                "age":
                age,

                "bmi":
                bmi,

                "glucose":
                avg_glucose_level,

                "age_group":
                age_group,

                "bmi_category":
                bmi_category,

                "risk_factor_count":
                risk_factor_count,

                "hypertension":
                hypertension,

                "heart_disease":
                heart_disease,

                "smoking_status":
                smoking_status
            }


            save_prediction_history(
                age,
                bmi,
                avg_glucose_level,
                probability,
                risk_level,
                prediction
            )


            st.success(
                "✅ Prediction generated successfully."
            )


            st.divider()


            # ------------------------------------------------
            # RESULT HEADER
            # ------------------------------------------------

            st.markdown("## 📊 Assessment Result")


            risk_class = get_risk_class(
                risk_level
            )

            risk_description = get_risk_description(
                risk_level
            )


            if risk_level == "LOW":

                icon = "🟢"

            elif risk_level == "MEDIUM":

                icon = "🟡"

            else:

                icon = "🔴"


            st.markdown(
                f"""
                <div class="{risk_class}">

                    <div class="risk-title">
                        {icon} {risk_level} RISK
                    </div>

                    <div class="risk-description">
                        {risk_description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.write("")


            # ------------------------------------------------
            # MAIN RESULT COLUMNS
            # ------------------------------------------------

            result1, result2 = st.columns(2)


            with result1:

                st.plotly_chart(
                    create_risk_gauge(probability),
                    use_container_width=True
                )


            with result2:

                st.markdown("### 📌 Prediction Summary")


                if prediction == 1:

                    st.error(
                        "🔴 Higher Stroke Risk"
                    )

                else:

                    st.success(
                        "🟢 Lower Stroke Risk"
                    )


                st.metric(
                    "Stroke Probability",
                    f"{probability:.2f}%"
                )


                st.metric(
                    "Risk Factor Count",
                    risk_factor_count
                )


                st.metric(
                    "Age Group",
                    age_group
                )


            # ------------------------------------------------
            # PATIENT ANALYSIS
            # ------------------------------------------------

            st.divider()

            st.markdown(
                "## 🔎 Patient Risk Analysis"
            )


            analysis1, analysis2, analysis3 = st.columns(3)


            with analysis1:

                st.info(
                    f"""
                    ### 👤 Age

                    **{age} years**

                    Group: **{age_group}**
                    """
                )


            with analysis2:

                st.info(
                    f"""
                    ### ⚖️ BMI

                    **{bmi:.1f}**

                    Category: **{bmi_category}**
                    """
                )


            with analysis3:

                st.info(
                    f"""
                    ### 🧪 Glucose

                    **{avg_glucose_level:.1f}**

                    Risk indicator based on
                    provided value.
                    """
                )


            # ------------------------------------------------
            # RISK FACTOR TABLE
            # ------------------------------------------------

            st.markdown(
                "## ⚠️ Risk Factor Details"
            )


            factor_data = {

                "Risk Factor": [

                    "Age",

                    "Hypertension",

                    "Heart Disease",

                    "Average Glucose",

                    "BMI",

                    "Smoking"
                ],

                "Patient Value": [

                    f"{age} years",

                    "Present"
                    if hypertension == 1
                    else "Not Present",

                    "Present"
                    if heart_disease == 1
                    else "Not Present",

                    f"{avg_glucose_level:.1f}",

                    f"{bmi:.1f}",

                    smoking_status
                ],

                "Indicator": [

                    "⚠️"
                    if age >= 60
                    else "🟢",

                    "🔴"
                    if hypertension == 1
                    else "🟢",

                    "🔴"
                    if heart_disease == 1
                    else "🟢",

                    "🔴"
                    if avg_glucose_level >= 140
                    else (
                        "🟡"
                        if avg_glucose_level >= 100
                        else "🟢"
                    ),

                    "🔴"
                    if bmi >= 30
                    else (
                        "🟡"
                        if bmi >= 25
                        else "🟢"
                    ),

                    "🔴"
                    if smoking_status == "smokes"
                    else (
                        "🟡"
                        if smoking_status == "formerly smoked"
                        else "🟢"
                    )
                ]
            }


            factor_df = pd.DataFrame(
                factor_data
            )


            st.dataframe(
                factor_df,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # EXPLAINABLE RESULT
            # ------------------------------------------------

            st.markdown(
                "## 🧠 Why Did the System Produce This Result?"
            )


            explanations = generate_explanation(
                age,
                bmi,
                avg_glucose_level,
                hypertension,
                heart_disease,
                smoking_status
            )


            for explanation in explanations:

                st.write(
                    explanation
                )


            st.caption(
                "The above explanation summarizes patient input "
                "characteristics. It is not a direct explanation "
                "of individual model coefficients or feature "
                "importance unless an explainability method such "
                "as SHAP is integrated."
            )


            # ------------------------------------------------
            # FACTOR CHART
            # ------------------------------------------------

            st.markdown(
                "## 📈 Patient Factor Visualization"
            )


            st.plotly_chart(
                create_risk_factor_chart(
                    age,
                    bmi,
                    avg_glucose_level,
                    hypertension,
                    heart_disease
                ),
                use_container_width=True
            )


            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            st.warning(
                """
                ⚠️ **Important:** This result is generated by a
                machine learning model for educational and
                decision-support purposes. It is not a medical
                diagnosis and should not replace evaluation by
                a qualified healthcare professional.
                """
            )


        except Exception as e:

            st.error(
                "❌ Prediction Error"
            )

            st.exception(e)


# ============================================================
# PAGE 3 - ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="section-title">📊 Prediction Analytics</div>',
        unsafe_allow_html=True
    )


    history = st.session_state.prediction_history


    if len(history) == 0:

        st.info(
            """
            📭 No prediction history is available yet.

            Go to **Risk Assessment** and perform a prediction
            first.
            """
        )

    else:

        history_df = pd.DataFrame(history)


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        total = len(history_df)

        high = len(
            history_df[
                history_df["Risk Level"] == "HIGH"
            ]
        )

        medium = len(
            history_df[
                history_df["Risk Level"] == "MEDIUM"
            ]
        )

        low = len(
            history_df[
                history_df["Risk Level"] == "LOW"
            ]
        )


        a1, a2, a3, a4 = st.columns(4)


        with a1:

            st.metric(
                "Total",
                total
            )


        with a2:

            st.metric(
                "🔴 High",
                high
            )


        with a3:

            st.metric(
                "🟡 Medium",
                medium
            )


        with a4:

            st.metric(
                "🟢 Low",
                low
            )


        st.divider()


        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        chart1, chart2 = st.columns(2)


        with chart1:

            risk_counts = pd.DataFrame({

                "Risk Level":
                [
                    "LOW",
                    "MEDIUM",
                    "HIGH"
                ],

                "Count":
                [
                    low,
                    medium,
                    high
                ]
            })


            fig = px.pie(
                risk_counts,
                names="Risk Level",
                values="Count",
                title="Risk Level Distribution",
                hole=0.45
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with chart2:

            fig2 = px.scatter(
                history_df,
                x="Age",
                y="Probability (%)",
                size="BMI",
                color="Risk Level",
                hover_data=[
                    "Glucose",
                    "Risk Level"
                ],
                title="Age vs Stroke Probability"
            )


            st.plotly_chart(
                fig2,
                use_container_width=True
            )


        # ----------------------------------------------------
        # PROBABILITY TREND
        # ----------------------------------------------------

        st.markdown(
            "### 📈 Prediction Probability Trend"
        )


        trend_df = history_df.copy()

        trend_df["Assessment"] = range(
            1,
            len(trend_df) + 1
        )


        fig3 = px.line(
            trend_df,
            x="Assessment",
            y="Probability (%)",
            markers=True,
            title="Stroke Probability Across Assessments"
        )


        fig3.update_yaxes(
            range=[0, 100]
        )


        st.plotly_chart(
            fig3,
            use_container_width=True
        )


        # ----------------------------------------------------
        # HISTORY TABLE
        # ----------------------------------------------------

        st.markdown(
            "### 📋 Prediction History"
        )


        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # DOWNLOAD HISTORY
        # ----------------------------------------------------

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="⬇️ Download Prediction History",
            data=csv_data,
            file_name="stroke_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


        # ----------------------------------------------------
        # CLEAR HISTORY
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Prediction History"
        ):

            st.session_state.prediction_history = []

            st.rerun()


# ============================================================
# PAGE 4 - BATCH PREDICTION
# ============================================================

elif page == "📁 Batch Prediction":

    st.markdown(
        '<div class="section-title">📁 Batch Stroke Prediction</div>',
        unsafe_allow_html=True
    )


    st.write(
        """
        Upload a CSV file containing multiple patient records.
        The application will generate predictions for all rows.
        """
    )


    # --------------------------------------------------------
    # EXPECTED COLUMNS
    # --------------------------------------------------------

    st.markdown(
        "### 📋 Required CSV Columns"
    )


    required_columns = [

        "gender",

        "age",

        "hypertension",

        "heart_disease",

        "ever_married",

        "work_type",

        "Residence_type",

        "avg_glucose_level",

        "bmi",

        "smoking_status"

    ]


    st.code(
        ", ".join(required_columns)
    )


    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "📤 Upload Patient CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )


            st.success(
                f"✅ File loaded successfully: "
                f"{len(batch_df)} records"
            )


            st.markdown(
                "### 👀 Uploaded Data"
            )


            st.dataframe(
                batch_df.head(20),
                use_container_width=True
            )


            missing_columns = [

                column

                for column in required_columns

                if column not in batch_df.columns
            ]


            if missing_columns:

                st.error(
                    "❌ Missing required columns:"
                )

                st.write(
                    missing_columns
                )


            else:

                if st.button(
                    "🚀 Run Batch Prediction",
                    type="primary",
                    use_container_width=True
                ):

                    try:

                        batch_input = batch_df[
                            required_columns
                        ].copy()


                        # ------------------------------------
                        # CREATE ENGINEERED FEATURES
                        # ------------------------------------

                        batch_input["age_group"] = (
                            batch_input["age"].apply(
                                calculate_age_group
                            )
                        )


                        batch_input["bmi_category"] = (
                            batch_input["bmi"].apply(
                                calculate_bmi_category
                            )
                        )


                        batch_input[
                            "risk_factor_count"
                        ] = (

                            batch_input[
                                "hypertension"
                            ].astype(int)

                            +

                            batch_input[
                                "heart_disease"
                            ].astype(int)

                            +

                            (
                                batch_input["age"]
                                >= 60
                            ).astype(int)

                            +

                            (
                                batch_input[
                                    "avg_glucose_level"
                                ]
                                >= 140
                            ).astype(int)
                        )


                        # ------------------------------------
                        # MODEL PREDICTION
                        # ------------------------------------

                        predictions = model.predict(
                            batch_input
                        )


                        if hasattr(
                            model,
                            "predict_proba"
                        ):

                            probabilities = (
                                model.predict_proba(
                                    batch_input
                                )[:, 1]
                                * 100
                            )

                        else:

                            probabilities = (
                                predictions
                                * 100
                            )


                        risk_levels = [

                            get_risk_level(
                                probability
                            )

                            for probability
                            in probabilities
                        ]


                        # ------------------------------------
                        # RESULT
                        # ------------------------------------

                        result_df = batch_df.copy()


                        result_df[
                            "Stroke Probability (%)"
                        ] = np.round(
                            probabilities,
                            2
                        )


                        result_df[
                            "Risk Level"
                        ] = risk_levels


                        result_df[
                            "Prediction"
                        ] = [

                            "Higher Stroke Risk"
                            if prediction == 1
                            else "Lower Stroke Risk"

                            for prediction
                            in predictions
                        ]


                        st.success(
                            "✅ Batch prediction completed."
                        )


                        # ------------------------------------
                        # RESULTS
                        # ------------------------------------

                        st.markdown(
                            "### 📊 Batch Prediction Results"
                        )


                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )


                        # ------------------------------------
                        # SUMMARY
                        # ------------------------------------

                        st.markdown(
                            "### 📈 Batch Summary"
                        )


                        b1, b2, b3, b4 = st.columns(4)


                        with b1:

                            st.metric(
                                "Total Patients",
                                len(result_df)
                            )


                        with b2:

                            st.metric(
                                "High Risk",
                                risk_levels.count(
                                    "HIGH"
                                )
                            )


                        with b3:

                            st.metric(
                                "Medium Risk",
                                risk_levels.count(
                                    "MEDIUM"
                                )
                            )


                        with b4:

                            st.metric(
                                "Low Risk",
                                risk_levels.count(
                                    "LOW"
                                )
                            )


                        # ------------------------------------
                        # BATCH CHART
                        # ------------------------------------

                        risk_summary = pd.DataFrame({

                            "Risk Level":
                            [
                                "LOW",
                                "MEDIUM",
                                "HIGH"
                            ],

                            "Patients":
                            [
                                risk_levels.count(
                                    "LOW"
                                ),

                                risk_levels.count(
                                    "MEDIUM"
                                ),

                                risk_levels.count(
                                    "HIGH"
                                )
                            ]
                        })


                        fig_batch = px.bar(
                            risk_summary,
                            x="Risk Level",
                            y="Patients",
                            title="Batch Risk Distribution"
                        )


                        st.plotly_chart(
                            fig_batch,
                            use_container_width=True
                        )


                        # ------------------------------------
                        # DOWNLOAD
                        # ------------------------------------

                        result_csv = (
                            result_df
                            .to_csv(index=False)
                            .encode("utf-8")
                        )


                        st.download_button(
                            label="⬇️ Download Batch Results",
                            data=result_csv,
                            file_name=(
                                "stroke_batch_predictions.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )


                    except Exception as e:

                        st.error(
                            "❌ Batch Prediction Error"
                        )

                        st.exception(e)


        except Exception as e:

            st.error(
                "❌ Unable to read CSV file."
            )

            st.exception(e)


# ============================================================
# PAGE 5 - ABOUT SYSTEM
# ============================================================

elif page == "ℹ️ About System":

    st.markdown(
        '<div class="section-title">ℹ️ About the System</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PROJECT DESCRIPTION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="custom-card">

        ## 🩺 Intelligent Stroke Risk Prediction &
        Clinical Decision Support System

        This project uses supervised machine learning to
        estimate stroke risk from patient demographic,
        lifestyle and health-related information.

        The system is designed as an educational and
        decision-support application.

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.markdown("## 🛠️ Technology Stack")


    tech1, tech2, tech3, tech4 = st.columns(4)


    with tech1:

        st.info(
            """
            ### 🐍 Python

            Programming language
            used for the ML pipeline
            and application.
            """
        )


    with tech2:

        st.info(
            """
            ### 🤖 Scikit-learn

            Machine learning models
            and preprocessing.
            """
        )


    with tech3:

        st.info(
            """
            ### 🎨 Streamlit

            Interactive web application
            interface.
            """
        )


    with tech4:

        st.info(
            """
            ### 📊 Plotly

            Interactive data
            visualizations.
            """
        )


    # --------------------------------------------------------
    # ML PIPELINE
    # --------------------------------------------------------

    st.markdown("## 🔄 Machine Learning Pipeline")


    pipeline_steps = [

        "📂 Dataset",

        "🧹 Data Cleaning",

        "🔍 Exploratory Data Analysis",

        "⚙️ Feature Engineering",

        "🔤 Encoding",

        "📏 Scaling",

        "⚖️ Imbalance Handling",

        "🤖 Model Training",

        "🎯 Model Evaluation",

        "💾 Pipeline Export",

        "🌐 Streamlit Deployment"
    ]


    for index, step in enumerate(
        pipeline_steps,
        start=1
    ):

        st.write(
            f"**{index}.** {step}"
        )


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    st.markdown("## 🤖 Models")


    model_col1, model_col2 = st.columns(2)


    with model_col1:

        st.success(
            """
            ### Logistic Regression

            A linear classification algorithm
            used as a baseline model for
            stroke-risk classification.
            """
        )


    with model_col2:

        st.success(
            """
            ### Random Forest

            An ensemble learning algorithm
            that combines multiple decision
            trees for classification.
            """
        )


    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    st.markdown("## 📋 Input Features")


    feature_df = pd.DataFrame({

        "Feature": [

            "Gender",

            "Age",

            "Hypertension",

            "Heart Disease",

            "Ever Married",

            "Work Type",

            "Residence Type",

            "Average Glucose Level",

            "BMI",

            "Smoking Status"
        ],

        "Type": [

            "Categorical",

            "Numerical",

            "Binary",

            "Binary",

            "Categorical",

            "Categorical",

            "Categorical",

            "Numerical",

            "Numerical",

            "Categorical"
        ]
    })


    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        """
        ⚠️ **Medical Disclaimer**

        This application is an academic machine learning
        project developed for educational and decision-support
        purposes.

        The prediction is generated from a trained machine
        learning model and should not be interpreted as a
        confirmed medical diagnosis.

        Users should consult qualified healthcare professionals
        for actual medical evaluation and treatment decisions.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🩺 Intelligent Stroke Risk Prediction System
    <br>
    AI-Based Healthcare Decision Support Project
    <br><br>
    Developed for Academic / Educational Purposes

    </div>
    """,
    unsafe_allow_html=True
)