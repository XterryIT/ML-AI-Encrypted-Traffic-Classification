# Encrypted Traffic Classification: Detecting C2 Tunnels in DoH

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Cybersecurity](https://img.shields.io/badge/Focus-Cybersecurity-red.svg)

## 📌 Overview
This project focuses on the detection of malicious DNS-over-HTTPS (DoH) traffic using Machine Learning. As threat actors increasingly use encrypted channels for Command and Control (C2) communication, traditional Deep Packet Inspection (DPI) becomes ineffective. This system classifies traffic based on flow statistics and time-series metadata without breaking encryption.

## 🔬 Methodology: CRISP-DM
The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:
1. **Business Understanding:** Identifying the risks of DoH tunneling (dns2tcp, DNSCat2, Iodine).
2. **Data Understanding:** Analyzing the `CIRA-CIC-DoHBrw-2020` dataset.
3. **Data Preparation:** Handling class imbalance using **SMOTE** and feature scaling.
4. **Modeling:** Training Random Forest, Decision Trees, and KNN classifiers.
5. **Evaluation:** Comparing models based on Precision, Recall, and F1-Score.

## 📊 Dataset Features
The model analyzes encrypted flows based on:
* **Duration:** Total time of the flow.
* **Packet Inter-arrival Time (IAT):** Statistical distribution of delays between packets.
* **Volumetric stats:** Byte count, packet length variance, and flow directionality.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (SMOTE), `matplotlib`, `seaborn`.
* **Environment:** Jupyter Notebook / Python Script.

## 📈 Key Results
The **Random Forest** classifier demonstrated the highest performance, especially after addressing the data imbalance.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **99.2%** | **0.99** | **0.99** | **0.99** |
| Decision Tree | 98.5% | 0.98 | 0.98 | 0.98 |
| KNN | 97.1% | 0.97 | 0.97 | 0.97 |

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/XterryIT/Encrypted-Traffic-Classification.git](https://github.com/XterryIT/Encrypted-Traffic-Classification.git)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```bash
   python main.py
   ```

## 📝 Authors
* **Andrii Ptashkohrai** - Student at WUST
* **Yana Baloshenko** - Student at WUST
