import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sklearn Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION ---
TRAIN_FILE = "train70_auto.csv"
TEST_FILE = "test30_auto.csv"
SEED = 7

print("Loading datasets...")
dftrain = pd.read_csv(TRAIN_FILE)
dftest = pd.read_csv(TEST_FILE)

# --- 2. PREPROCESSING (Corrected Dictionary Approach) ---
# We use a DICTIONARY to store a separate encoder for every column.
label_encoders = {} 

cat_columns = dftrain.select_dtypes(['object', 'category']).columns

print("Encoding categorical columns...")
for col in cat_columns:
    # 1. Create a NEW encoder for this specific column
    encoder = LabelEncoder()
    
    # 2. Fit on combined data
    combined_data = pd.concat([dftrain[col], dftest[col]], axis=0).astype(str)
    encoder.fit(combined_data)
    
    # 3. Store the encoder in our dictionary
    label_encoders[col] = encoder
    
    # 4. Transform the data
    dftrain[col] = encoder.transform(dftrain[col].astype(str))
    dftest[col] = encoder.transform(dftest[col].astype(str))

# Separate Features (X) and Target (y)
x_columns = dftrain.columns.drop('target')
print("Saving column order...")
joblib.dump(x_columns, 'column_names.pkl')
x_train = dftrain[x_columns].values
y_train = dftrain['target']

x_test = dftest[x_columns].values
y_test = dftest['target']

print(f"Data Loaded. Train Shape: {x_train.shape}, Test Shape: {x_test.shape}")

# --- 3. MODEL TRAINING ---
print("\nStarting Random Forest Training...")

classifier = RandomForestClassifier(
    n_estimators=100, 
    random_state=SEED, 
    verbose=0, 
    n_jobs=-1 
)

# Start Timer
start_train = time.time()

# Train
classifier.fit(x_train, y_train)

# --- SAVE FOR PRODUCTION ---
print("\nSaving model to file...")

# Save the Classifier
joblib.dump(classifier, 'rf_ids_model.pkl')

# Save the Dictionary of Encoders
# Now you save ALL encodings, not just the last one
joblib.dump(label_encoders, 'label_encoder_dict.pkl')

print("Model and Encoders saved successfully!")

# End Timer
end_train = time.time()
train_time = end_train - start_train
print(f"Training Time: {train_time:.4f} seconds")

# --- 4. PREDICTION & EVALUATION ---
print("\nRunning Predictions on Test Set...")

start_test = time.time()
y_pred = classifier.predict(x_test)
end_test = time.time()
test_time = end_test - start_test

# FIXED TYPO HERE: used 'test_time' instead of 'test_test_time'
print(f"Test Time: {test_time:.4f} seconds")

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("-" * 30)
print(f"ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 SCORE: {f1:.4f}")
print("-" * 30)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Detailed Report with Actual Names
print("\nClassification Report:\n")

# 1. Retrieve the encoder for the 'target' column
target_encoder = label_encoders['target']

# 2. Get the list of original names (e.g., ['bruteforce', 'legitimate', 'slowite'])
# We convert them to strings just to be safe
target_names = [str(cls) for cls in target_encoder.classes_]

# 3. Generate the report using the names
print(classification_report(y_test, y_pred, target_names=target_names))

# --- 5. FEATURE IMPORTANCE ---
print("\n--- Feature Importance Analysis ---")
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 10 Features contributing to detection:")
for i in range(10):
    if i < len(x_columns):
        print(f"{i+1}. {x_columns[indices[i]]}: {importances[indices[i]]:.6f}")