# doctor.py
import streamlit as st
import joblib
import pandas as pd

# Load components
model = joblib.load('../models/model.pkl')
mlb = joblib.load('../models/symptom_encoder.pkl')
le = joblib.load('../models/label_encoder.pkl')

# Load treatments
with open('treatment_db.json') as f:
    treatments = pd.read_json(f).to_dict()

# App interface
st.set_page_config(page_title="AI Doctor", layout="wide")
st.title("🩺 AI Disease Predictor")

# Symptom selector
selected = st.multiselect(
    "Select your symptoms:",
    options=mlb.classes_,
    placeholder="Choose from list..."
)

if st.button("Diagnose"):
    if not selected:
        st.error("Please select at least 1 symptom!")
    else:
        # Encode symptoms
        encoded = mlb.transform([selected])
        
        # Predict
        proba = model.predict_proba(encoded)[0]
        top3 = sorted(zip(proba, le.classes_), reverse=True)[:3]
        
        # Display results
        st.success("Diagnosis Results:")
        cols = st.columns(3)
        
        for idx, (prob, disease) in enumerate(top3):
            with cols[idx]:
                st.subheader(f"#{idx+1}: {disease}")
                st.metric("Confidence", f"{prob*100:.1f}%")
                
                if disease in treatments:
                    st.write("**Medications:**")
                    st.write(", ".join(treatments[disease]['medications']))
                    st.write("**Advice:**")
                    st.write(treatments[disease]['advice'])
                else:
                    st.warning("Treatment info not available")