# processes_data.py
import pandas as pd
import glob
import os

# Combine all CSVs
all_files = glob.glob("data/raw/*.csv")
combined = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

# Clean data
combined = combined.apply(lambda x: x.str.strip().replace('', pd.NA))
combined = combined.dropna(how='all', subset=['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4'])

# Save processed data
os.makedirs("data/processed", exist_ok=True)
combined.to_csv("data/processed/full_data.csv", index=False)