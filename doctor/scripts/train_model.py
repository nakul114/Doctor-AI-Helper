# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import joblib

# Load processed data
df = pd.read_csv("data/processed/full_data.csv")

# Prepare symptoms
symptoms = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']]\
    .apply(lambda row: [s for s in row if pd.notna(s)], axis=1)

# Create encoders
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptoms)

le = LabelEncoder()
y = le.fit_transform(df['Disease'])

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save components
joblib.dump(model, 'models/model.pkl')
joblib.dump(mlb, 'models/symptom_encoder.pkl')
joblib.dump(le, 'models/label_encoder.pkl')