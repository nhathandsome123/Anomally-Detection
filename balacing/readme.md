Here is a specialized README.md for your data preprocessing module. It clearly explains the folder structure and the logic behind the balancing script.

Markdown

# Data Preprocessing & Balancing Module

This module is responsible for preparing the raw network traffic data for Machine Learning. It converts raw CSV logs into a perfectly balanced, clean, and split dataset ready for training.

## 📂 Directory Structure

Ensure your folders are organized as follows before running the script:

```text
/
├── balancer.py                  # The main preprocessing script
├── original_data/               # [INPUT] Raw CSV files from Wireshark/InfluxDB
│   ├── mqtt_hr_cleaned.csv      # Normal/Legitimate traffic
│   ├── slowite_reordered.csv    # SlowITe Attack traffic
│   ├── slow_bruteforce...csv    # Slow Brute Force traffic
│   └── rotating_bruteforce...csv# Rotating Brute Force traffic
└── balanced_data/               # [OUTPUT] The script saves files here
    ├── train70_auto.csv         # 70% of data for training
    └── test30_auto.csv          # 30% of data for testing
🛠️ The Script: balancer.py
What it does
Cleaning: Automatically removes non-behavioral features that cause overfitting (IP addresses, MAC addresses, timestamps, payloads, etc.).

Balancing (Upsampling):

It calculates the total amount of Legitimate traffic.

It forces the Attack traffic to match that count (50/50 split).

It achieves this by Upsampling with Replacement: taking the small attack files and randomly sampling rows from them until they reach the target size.

Splitting: Automatically shuffles and splits the final combined dataset into 70% Training and 30% Testing.

How to Run
Make sure your raw files are in the original_data folder.

Run the script:

Bash

python balancer.py
The script will generate train70_auto.csv and test30_auto.csv inside the balanced_data folder (or root, depending on script config).

📊 Logic Details
Input Data:

Legitimate: Uses mqtt_hr_cleaned.csv.

Attacks: Uses specific attack CSVs defined in the script configuration.

Cleaning Rules:

Removed: ip.src, ip.dst, frame.time, mqtt.topic, payload_raw, etc.

Retained: Behavioral metrics like bytes_toserver, flags, duration.

Output Balance:

The final dataset will contain roughly 50% Normal Traffic and 50% Attack Traffic (divided equally among all attack types).



