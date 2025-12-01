# MQTT Intrusion Detection System (IDS) using Random Forest

This project implements a Machine Learning-based Intrusion Detection System designed to detect attacks on MQTT IoT networks. It uses a Random Forest classifier to distinguish between legitimate traffic and various attacks (SlowITe, Brute Force, etc.).

## 📂 File Descriptions

### 1. Code Scripts
* **`RandomForest.py`** (The "Teacher"):
    * **Purpose:** This is the main training script.
    * **What it does:** It loads the training data (`train70_auto.csv`), preprocesses it, trains the Random Forest model, and then testing using the testing data ('test30_auto.csv') calculates performance metrics (Accuracy, F1-Score, Confusion Matrix), and saves the trained model artifacts for production.
* **`production.py`** (The "Detector"):
    * **Purpose:** This is the deployment script for real-world testing.
    * **What it does:** It loads the saved model and encoders, reads a new raw CSV file (simulating live traffic), cleans/encodes it exactly like the training data, and outputs a prediction report (`ids_scan_report.csv`) identifying attacks.

### 2. Model Artifacts (The "Brain")
* **`rf_ids_model.pkl`**: The trained Random Forest Classifier model itself.
* **`label_encoder_dict.pkl`**: A dictionary containing the specific encodings for categorical features (e.g., mapping "TCP" -> 1). This ensures the production script speaks the same language as the model.
* **`column_names.pkl`**: A list of the exact feature order used during training. This prevents crashes if the production input file has scrambled columns.


