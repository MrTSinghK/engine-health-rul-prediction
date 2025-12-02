# 🚀 Engine Health RUL Prediction using Deep Learning  
### Predicting Remaining Useful Life (RUL) of Turbofan Engines (CMAPSS FD001–FD004)

**Author:** Tanvir Harpreet Singh Kohli  
**Tech Stack:** Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Scikit-Learn  
**Dataset:** NASA CMAPSS Turbofan Engine Degradation Dataset

---

## 📌 Overview

This project develops a complete **AI-powered Engine Health Monitoring System** capable of predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines.  
The system uses the **NASA CMAPSS dataset (FD001–FD004)** and a **CNN-BiLSTM-Attention** hybrid architecture to model both short-term sensor patterns and long-range engine degradation behavior.

The final model achieves strong predictive performance across all datasets and is packaged into a fully reproducible notebook pipeline.

---

## 📂 Project Structure
- **data/** — Raw CMAPSS datasets (NOT included in repo)
- **notebooks/**
  - `Engine Health Prediction - Tanvir.ipynb` — main notebook (preprocessing, training, eval)
- **artifacts/** — model checkpoints & outputs (gitignored)
- **requirements.txt** — Python package list
- **.gitignore** — rules to avoid committing large files
- **README.md** — this file

---

## 🧠 Model Architecture

A hybrid deep learning architecture was used:

### **1️⃣ Convolutional Layers (CNN)**
- Extract short-term temporal sensor patterns  
- Reduce noise  
- Capture early degradation signals  

### **2️⃣ BiLSTM Layers**
- Learn long-range temporal dependencies  
- Understand engine wear trends  
- Model bidirectional behavior  

### **3️⃣ Attention Mechanism**
- Identifies **critical time steps** that contribute most to degradation  
- Improves interpretability and prediction quality  

### **4️⃣ Dense Output Layer**
- Outputs final RUL  
- Includes L2 regularization for stability  

---

## ⚙️ Methodology

### ✔ Data Preparation  
- Sliding window of **W = 40 cycles**  
- Sequence generation per engine unit  
- Z-score normalization  
- Train/validation split using **GroupShuffleSplit**  
- Test windows generated from the final W cycles of each engine  

### ✔ Training Configuration  
- **Optimizer:** Adam  
- **Loss:** MAE  
- **Metrics:** RMSE, PHM Score  
- **Callbacks:** EarlyStopping + ReduceLROnPlateau + Checkpointing  
- Training done **separately for FD001–FD004**  
- Best model per FD stored automatically in `artifacts/`

---

## 📊 Model Performance (Test Set)

| Dataset | MAE ↓ | RMSE ↓ | PHM Score ↓ | Units |
|--------|-------|--------|-------------|--------|
| FD001  | *13.9* | *18.9* | *1337*     | *100* |
| FD002  | *28.2* | *39.1* | *394366*   | *259* |
| FD003  | *11.3* | *15.5* | *516*      | *100* |
| FD004  | *31.7* | *38.5* | *37499*    | *248* |


---

## 📈 Visual Results to Add
- True vs Predicted RUL
  <img width="620" height="622" alt="FD004 Regression Fit Line" src="https://github.com/user-attachments/assets/912362e4-8018-4f89-95f0-c0f8b6f9fe9f" />

- Error Distribution
  <img width="463" height="468" alt="VAL vs MEA" src="https://github.com/user-attachments/assets/41bed82d-5350-41c5-a3a3-7181f200698b" />
  
- Training/validation loss curves
  <img width="532" height="314" alt="MAE CURVES" src="https://github.com/user-attachments/assets/1a1e19f5-d9c8-41d6-afbf-4eb5258b88ff" />

- FD001 True vs Predicted Plot
<img width="1189" height="490" alt="FD001 True vs Predicted Plot" src="https://github.com/user-attachments/assets/fe0b3db0-bd7c-4c14-bc8e-b7457a65b332" />

- FD002 True vs Predicted Plot
<img width="1188" height="490" alt="FD002 True vs Predicted Plot" src="https://github.com/user-attachments/assets/d53a6d13-0cbd-4574-b74e-9cc7b80cc02c" />

- FD003 True vs Predicted Plot
<img width="1189" height="490" alt="FD003 True vs Predicted Plot" src="https://github.com/user-attachments/assets/66e8f8b2-d935-4ef5-9599-f056bf7d92a7" />

- FD004 True vs Predicted Plot
<img width="1189" height="490" alt="FD004 True vs Predicted Plot" src="https://github.com/user-attachments/assets/01ec18b3-0fd9-4f90-b348-c1bc07f16594" />


---

## 🛠️ Installation & Setup
  clone_repository:
    - "git clone https://github.com/MrTSinghK/engine-health-rul-prediction.git"
    - "cd engine-health-rul-prediction"
  install_dependencies:
    - "pip install -r requirements.txt"
  launch_jupyter:
    - "jupyter notebook"
  important_note:
    disable_retraining: "Set RUN_TRAINING = False inside the notebook to avoid long training times."

⭐ Key Highlights:
  - "End-to-end predictive maintenance pipeline"
  - "CNN-BiLSTM-Attention hybrid model"
  - "Aerospace-grade PHM scoring"
  - "Clean diagnostics and visualizations"
  - "Optimized for Airbus / Rolls-Royce / GE Aviation interviews"
  - "Fully reproducible without retraining"

🧩 Future Improvements:
  - "Transformer-based RUL prediction"
  - "SHAP-based sensor interpretability"
  - "Real-time RUL monitoring dashboard"
  - "TensorFlow Lite deployment"
  - "Uncertainty-aware predictions"

🏆 Acknowledgements:
  - "NASA Prognostics Center – CMAPSS Dataset"
  - "PHM Society scoring system"
  - "TensorFlow, NumPy, Pandas, Scikit-Learn open-source communities"

📬 Contact:
  Name: "Tanvir Harpreet Singh Kohli"
  
  LinkedIn: "(https://www.linkedin.com/in/tanvir-harpreet-singh-kohli-a45716219/)"
  
  Email: "kohlitanvirsingh@gmail.com"





