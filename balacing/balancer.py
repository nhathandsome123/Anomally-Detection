import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# --- CONFIGURATION ---
LEGIT_FILE = './original_data/mqtt_hr_cleaned.csv'
ATTACK_FILES = {
    'slowite': './original_data/slowite_reordered.csv',
    'slow_brute_force': './original_data/slow_bruteforce_reordered.csv',
    'rotating_brute_force': './original_data/rotating_bruteforce_reordered.csv',
}
SEED = 7

# Reuse your existing clean function
def cleandata(df):
    # (Paste your existing deletion list here)
    columns_to_delete = [
        '_start', '_stop', '_time', '_measurement',
        'src_ip', 'dest_ip', 'src_port', 'dest_port',
        'client_id', 'client_identifier',
        'username', 'password', 'topic', 'payload_raw' 
        # ... Add all other columns from your original script ...
    ]
    # "ignore" errors means it won't crash if a column is missing in a new file
    df.drop(columns=columns_to_delete, errors='ignore', inplace=True)
    return df

print("--- Starting Automated Processing ---")

# 1. Load and Process Legitimate Data (The Baseline)
print(f"Loading {LEGIT_FILE}...")
df_legit = pd.read_csv(LEGIT_FILE)
df_legit.fillna(0, inplace=True)
df_legit['target'] = 'legitimate'
cleandata(df_legit)

# Automated Limit: Use ALL legitimate data available (or cap it if it's too huge)
# In your previous code, you capped it at 10 million. Let's keep that safety cap.
MAX_LEGIT_ROWS = 10000000 
if len(df_legit) > MAX_LEGIT_ROWS:
    df_legit = df_legit.head(MAX_LEGIT_ROWS)

count_legit = len(df_legit)
print(f"Legitimate Traffic Count: {count_legit} rows")

# 2. Calculate Targets for Attacks (The Logic)
# Goal: 50% Legitimate, 50% Attacks
# Total Attack Rows needed = Total Legitimate Rows
total_attack_target = count_legit
number_of_attack_types = len(ATTACK_FILES)

# Each attack type gets an equal slice of the pie
rows_per_attack = int(total_attack_target / number_of_attack_types)
print(f"Balancing Logic: {total_attack_target} total attack rows needed.")
print(f"Target per Attack File: {rows_per_attack} rows")

# 3. Process and Augment Attacks
all_attacks = []

for attack_name, filename in ATTACK_FILES.items():
    print(f"Processing {attack_name} ({filename})...")
    
    # Load
    df_attack = pd.read_csv(filename)
    df_attack.fillna(0, inplace=True)
    df_attack['target'] = attack_name
    cleandata(df_attack)
    
    # --- THE MAGIC LINE (Upsampling) ---
    # This replaces the 'for i in range(250)' loop.
    # It automatically samples existing rows until it reaches 'rows_per_attack'.
    # replace=True allows it to pick the same row multiple times (Upsampling).
    df_augmented = df_attack.sample(n=rows_per_attack, replace=True, random_state=SEED)
    
    all_attacks.append(df_augmented)
    print(f" -> Upsampled from {len(df_attack)} to {len(df_augmented)}")

# 4. Combine Everything
print("Combining datasets...")
df_total_attacks = pd.concat(all_attacks)
df_final = pd.concat([df_legit, df_total_attacks])

# Shuffle to mix Legitimate and Attacks thoroughly
df_final = shuffle(df_final, random_state=SEED)

print(f"Final Dataset Shape: {df_final.shape} (Should be roughly 2x Legitimate)")

# 5. Automated Split and Save
print("Splitting into Train (70%) and Test (30%)...")
train, test = train_test_split(df_final, test_size=0.3, random_state=SEED)

print("Saving to CSV...")
train.to_csv('train70_auto.csv', index=False)
test.to_csv('test30_auto.csv', index=False)

print("Done! Process Complete.")