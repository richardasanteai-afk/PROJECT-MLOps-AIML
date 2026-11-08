import os
import streamlit as st
import pandas as pd
import joblib

import os
import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Custom CSS for a cleaner look
# -------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #5A6A7A;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        height: 3rem;
        border-radius: 8px;
    }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #A5D6A7;
    }
    .warning-box {
        background-color: #FFF3E0;
        color: #E65100;
        border: 1px solid #FFCC80;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<p class="main-header">✈️ Tourism Package Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter customer details to predict the likelihood of purchasing a travel package</p>', unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# Input Form - Organised in Columns & Sections
# -------------------------------------------------
with st.form("prediction_form"):

    # ---- Section 1: Personal Information ----
    st.subheader("👤 Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
        Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
        MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])

    with col2:
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Monthly Income (₹)", min_value=1000.0, value=30000.0, step=500.0)

    with col3:
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        Passport = st.selectbox("Has Passport?", ["Yes", "No"])
        OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

    st.markdown("---")

    # ---- Section 2: Travel Preferences ----
    st.subheader("🧳 Travel Preferences")
    col4, col5, col6 = st.columns(3)

    with col4:
        NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2, step=1)
        NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=1, step=1)

    with col5:
        NumberOfTrips = st.number_input("Number of Trips (yearly)", min_value=1, max_value=20, value=3, step=1)
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

    with col6:
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

    st.markdown("---")

    # ---- Section 3: Sales Interaction ----
    st.subheader("📞 Sales Interaction Details")
    col7, col8, col9 = st.columns(3)

    with col7:
        DurationOfPitch = st.number_input("Duration of Pitch (mins)", min_value=0, max_value=100, value=15, step=1)

    with col8:
        NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3, step=1)

    with col9:
        PitchSatisfactionScore = st.select_slider(
            "Pitch Satisfaction Score",
            options=[1, 2, 3, 4, 5],
            value=3
        )

    st.markdown("")
    submitted = st.form_submit_button("🚀 Predict Purchase Likelihood")

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        'Age': Age,
        'TypeofContact': TypeofContact,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': Occupation,
        'Gender': Gender,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': ProductPitched,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus,
        'NumberOfTrips': NumberOfTrips,
        'Passport': 1 if Passport == "Yes" else 0,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': 1 if OwnCar == "Yes" else 0,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': Designation,
        'MonthlyIncome': MonthlyIncome
    }])

    classification_threshold = 0.35
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    # Probability metric
    st.metric(label="Purchase Probability", value=f"{prob:.1%}")

    # Result message
    if pred == 1:
        st.markdown(
            f'<div class="result-box success-box">✅ Customer is likely to purchase the travel package</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box warning-box">⚠️ Customer is unlikely to purchase the travel package</div>',
            unsafe_allow_html=True
        )

    # Additional insight
    st.caption(f"Classification threshold used: {classification_threshold} | Model confidence: {prob:.3f}")
