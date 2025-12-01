import joblib
import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION ---
# Change this to your actual test file
INPUT_CSV = "slowite_reordered.csv" 
MODEL_FILE = "rf_ids_model.pkl"
ENCODER_FILE = "label_encoder_dict.pkl"

# Columns to remove (Must match your training logic EXACTLY)
DROP_COLUMNS = [
    'frame.time_epoch', 'frame.time_relative', 'frame.time_delta',
    'ip.src', 'ip.dst', 'eth.src', 'eth.dst', 'tcp.srcport', 'tcp.dstport',
    'mqtt.topic', 'mqtt.clientid', 'payload_raw', 'payload_len', 'message_id',
    '_start', '_stop', '_time', '_measurement', 'client_identifier', 'username', 'password' 
]

def cleandata(df):
    # Only drop columns that actually exist in the file (errors='ignore')
    df_clean = df.drop(columns=DROP_COLUMNS, errors='ignore')
    return df_clean

# --- 2. LOAD SYSTEM ---
print(f"Loading Model from {MODEL_FILE}...")
if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
    print("ERROR: Model files not found. Please run 'RandomForest.py' first.")
    exit()

model = joblib.load(MODEL_FILE)
encoder_dict = joblib.load(ENCODER_FILE)
COLUMN_NAMES_FILE = "column_names.pkl"
if not os.path.exists(COLUMN_NAMES_FILE):
    print("ERROR: column_names.pkl not found. Re-run RandomForest.py.")
    exit()
model_columns = joblib.load(COLUMN_NAMES_FILE)
print("Column order loaded.")

print("System loaded successfully.")

# --- 3. LOAD & CLEAN TRAFFIC ---
print(f"Reading traffic from {INPUT_CSV}...")
try:
    df_input = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"ERROR: Could not find {INPUT_CSV}")
    exit()

print(f"Raw Traffic: {df_input.shape}")

# Pre-processing 1: Clean unwanted columns
df_features = cleandata(df_input.copy())

# Pre-processing 2: Remove 'target' if it exists (we are predicting it, not reading it)
ground_truth = None
if 'target' in df_features.columns:
    ground_truth = df_features['target'] # Save for accuracy check later
    df_features = df_features.drop(columns=['target'])

print(f"Features ready for analysis: {df_features.shape}")

# --- 4. ENCODING (The Production Way) ---
print("Encoding features...")

# We need to loop through the encoders we saved
for col, encoder in encoder_dict.items():
    if col == 'target': continue # Skip the target encoder
    
    if col in df_features.columns:
        # Convert to string to match training format
        df_features[col] = df_features[col].astype(str)
        
        # KEY PRODUCTION STEP: Handle "Unknown" categories
        # If the CSV has a value like "NewProtocol" that wasn't in training,
        # replace it with the first known class (or 0) to prevent a crash.
        known_classes = set(encoder.classes_)
        sorted_classes = sorted(list(known_classes))
        df_features[col] = df_features[col].apply(lambda x: x if x in known_classes else sorted_classes[0])
        # Transform
        df_features[col] = encoder.transform(df_features[col])
df_features = df_features.reindex(columns=model_columns, fill_value=0)
print("Columns reordered to match training data.")
# Fill NaN with 0 (Just in case)
df_features.fillna(0, inplace=True)

# --- 5. PREDICT ---
print("Running IDS Scan...")
try:
    # 1. Predict Class IDs (0, 1, 2...)
    pred_ids = model.predict(df_features)
    
    # 2. Convert IDs back to Names (legitimate, slowite...)
    target_encoder = encoder_dict['target']
    pred_names = target_encoder.inverse_transform(pred_ids)
    
    # Add predictions to the DataFrame for viewing
    results = df_input.copy()
    results['IDS_PREDICTION'] = pred_names
    
except Exception as e:
    print(f"Prediction Error: {e}")
    print("Double check that your CSV columns match the training features.")
    exit()

# --- 6. REPORT ---
print("\n" + "="*30)
print("       SCAN RESULTS       ")
print("="*30)

total_packets = len(results)
attacks_detected = results[results['IDS_PREDICTION'] != 'legitimate']
count_attacks = len(attacks_detected)
count_legit = total_packets - count_attacks

print(f"Total Packets Scanned: {total_packets}")
print(f"Normal Traffic:        {count_legit} ({(count_legit/total_packets)*100:.1f}%)")
print(f"Attacks Detected:      {count_attacks} ({(count_attacks/total_packets)*100:.1f}%)")

if count_attacks > 0:
    print("\n[!] ALERT: Attack Breakdown:")
    print(attacks_detected['IDS_PREDICTION'].value_counts())
else:
    print("\n[OK] System Clean. No threats detected.")

# Optional: Check Accuracy if we knew the answer
if ground_truth is not None:
    print("\n--- Self-Check Accuracy ---")
    correct = np.sum(results['IDS_PREDICTION'] == ground_truth)
    print(f"Correctly identified: {correct}/{total_packets} ({(correct/total_packets)*100:.2f}%)")

# Save detailed log
results.to_csv("ids_scan_slowite.csv", index=False)
print("\nDetailed report saved to 'ids_scan_slowite.csv'")