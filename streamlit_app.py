import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "titanic_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# Title and description
st.title("🚢 Titanic Survival Predictor")
st.markdown("""
This app predicts whether a passenger would have survived the Titanic disaster 
based on their characteristics.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: f"{x}st Class" if x == 1 else f"{x}nd Class" if x == 2 else f"{x}rd Class"
    )
    
    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.1
    )
    
    sibsp = st.number_input(
        "Number of Siblings/Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0
    )
    
    parch = st.number_input(
        "Number of Parents/Children Aboard",
        min_value=0,
        max_value=10,
        value=0
    )

with col2:
    fare = st.number_input(
        "Fare (in £)",
        min_value=0.0,
        max_value=600.0,
        value=50.0,
        step=0.01
    )
    
    gender = st.radio(
        "Gender",
        options=["Female", "Male"]
    )
    gender_male = 1 if gender == "Male" else 0
    
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["Southampton", "Cherbourg", "Queenstown"]
    )
    embarked_q = 1 if embarked == "Queenstown" else 0
    embarked_s = 1 if embarked == "Southampton" else 0

# Prediction button
if st.button("Predict Survival", type="primary", use_container_width=True):
    # Create feature array
    features = np.array([[pclass, age, sibsp, parch, fare, gender_male, embarked_q, embarked_s]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.success(f"✅ **Survived** with {probability[1]*100:.2f}% confidence")
    else:
        st.error(f"❌ **Did Not Survive** with {probability[0]*100:.2f}% confidence")
    
    # Show feature values
    with st.expander("View Input Features"):
        st.json({
            "Pclass": int(pclass),
            "Age": float(age),
            "SibSp": int(sibsp),
            "Parch": int(parch),
            "Fare": float(fare),
            "Gender_male": int(gender_male),
            "Embarked_Q": int(embarked_q),
            "Embarked_S": int(embarked_s)
        })

# Footer
st.markdown("---")
st.info("ℹ️ This model is trained on historical Titanic passenger data and is for educational purposes only.")