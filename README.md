# 🚖 PUMIS: Privacy-Preserving Urban Mobility Information System

**PUMIS** is a Context-Aware Differentially Private Framework for Synthetic Urban Mobility Data Generation. It is designed to solve the fundamental "Privacy-Utility-Realism" trilemma in smart city data sharing.

By leveraging Generative Adversarial Networks (ContextGAN), Differentially Private Stochastic Gradient Descent (DP-SGD), and a novel Physics-Informed Loss module, PUMIS generates high-fidelity synthetic mobility datasets that strictly adhere to real-world physics while providing formal mathematical guarantees against identity leakage.

## ✨ Key Features

* 🔒 **Cryptographic Privacy (DP-SGD):** Implements gradient clipping and Gaussian noise injection to guarantee ($\epsilon$, $\delta$)-Differential Privacy, making it immune to Membership Inference Attacks (MIA).

* 🧠 **Physics-Informed Neural Networks (**$L_{phys}$**):** Actively penalizes the Generator for breaking physical laws ($v = d/t$), reducing spatiotemporally impossible "Ghost Trips" from 18.4% (standard GANs) down to **0.2%**.

* 🔄 **Universal Auto-Synthesizer:** A dynamic, schema-agnostic engine that automatically profiles any uploaded numerical dataset, resizes the neural architecture, and trains a custom clone on the fly.

* 🛡️ **100% Data Sovereignty:** Designed as a localized application. No raw data, weights, or embeddings ever leave the offline environment, ensuring compliance with GDPR and HIPAA.

## 📊 Performance Metrics

Our evaluation demonstrates that PUMIS successfully balances utility and privacy:

| Metric Category | Evaluation Test | PUMIS Score | Baseline CTGAN | 
 | ----- | ----- | ----- | ----- | 
| **Statistical Utility** | Earth Mover's Distance (EMD) | **0.12** | 0.15 | 
| **Physical Realism** | "Ghost Trip" Violation Rate | **0.2%** | 18.4% | 
| **Privacy / Security** | Nearest Neighbor Adversarial Accuracy | **0.52** | \~0.80 (Vulnerable) | 
| **ML Efficacy (TSTR)** | Predictive Utility Drop ($R^2$) | **7.59%** | N/A | 

*(Note: An NNAA score of \~0.5 indicates random guessing by an attacker, proving immunity to identity inference).*

## 🚀 Installation & Setup

### 1. Prerequisites

Ensure you have Python 3.10+ installed on your machine. We recommend using a virtual environment.

### 2. Clone the Repository

`git clone https://github.com/yourusername/PUMIS.git
cd PUMIS`

### 3. Install Dependencies

`pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn streamlit pyarrow`

## 💻 Usage

PUMIS features two primary interfaces: an interactive web dashboard for data synthesis and a CLI evaluation script.

### 1. Launching the Auto-Synthesizer (Streamlit UI)

To launch the secure offline web prototype, run:
`streamlit run dpumis.py`

* **Tab 1 (NYC Pre-trained Engine):** Generate context-conditioned synthetic trips based on pre-trained NYC TLC weights.

* **Tab 2 (Universal Auto-Synthesizer):** Upload your own proprietary CSV to dynamically train a private GAN and download a synthetic clone.

### 2. Running the Machine Learning Efficacy Test (TSTR)

To mathematically validate the utility of your synthetic data (Train on Synthetic, Test on Real), ensure your real and synthetic CSVs are in the root directory and run:

`python tstr_evaluator.py`

This script will output the $R^2$ scores, Mean Absolute Error (MAE), and generate a comparison bar chart.

## 📂 Repository Structure

Here is the raw Markdown content for the README from the Canvas, ready for you to copy and paste.

Markdown
# 🚖 PUMIS: Privacy-Preserving Urban Mobility Information System

**PUMIS** is a Context-Aware Differentially Private Framework for Synthetic Urban Mobility Data Generation. It is designed to solve the fundamental "Privacy-Utility-Realism" trilemma in smart city data sharing.

By leveraging Generative Adversarial Networks (ContextGAN), Differentially Private Stochastic Gradient Descent (DP-SGD), and a novel Physics-Informed Loss module, PUMIS generates high-fidelity synthetic mobility datasets that strictly adhere to real-world physics while providing formal mathematical guarantees against identity leakage.

## ✨ Key Features

* 🔒 **Cryptographic Privacy (DP-SGD):** Implements gradient clipping and Gaussian noise injection to guarantee ($\epsilon$, $\delta$)-Differential Privacy, making it immune to Membership Inference Attacks (MIA).

* 🧠 **Physics-Informed Neural Networks (**$L_{phys}$**):** Actively penalizes the Generator for breaking physical laws ($v = d/t$), reducing spatiotemporally impossible "Ghost Trips" from 18.4% (standard GANs) down to **0.2%**.

* 🔄 **Universal Auto-Synthesizer:** A dynamic, schema-agnostic engine that automatically profiles any uploaded numerical dataset, resizes the neural architecture, and trains a custom clone on the fly.

* 🛡️ **100% Data Sovereignty:** Designed as a localized application. No raw data, weights, or embeddings ever leave the offline environment, ensuring compliance with GDPR and HIPAA.

## 📊 Performance Metrics

Our evaluation demonstrates that PUMIS successfully balances utility and privacy:

| Metric Category | Evaluation Test | PUMIS Score | Baseline CTGAN | 
 | ----- | ----- | ----- | ----- | 
| **Statistical Utility** | Earth Mover's Distance (EMD) | **0.12** | 0.15 | 
| **Physical Realism** | "Ghost Trip" Violation Rate | **0.2%** | 18.4% | 
| **Privacy / Security** | Nearest Neighbor Adversarial Accuracy | **0.52** | \~0.80 (Vulnerable) | 
| **ML Efficacy (TSTR)** | Predictive Utility Drop ($R^2$) | **7.59%** | N/A | 

*(Note: An NNAA score of \~0.5 indicates random guessing by an attacker, proving immunity to identity inference).*

## 🚀 Installation & Setup

### 1. Prerequisites

Ensure you have Python 3.10+ installed on your machine. We recommend using a virtual environment.

### 2. Clone the Repository

git clone https://github.com/yourusername/PUMIS.git
cd PUMIS


### 3. Install Dependencies

pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn streamlit pyarrow


## 💻 Usage

PUMIS features two primary interfaces: an interactive web dashboard for data synthesis and a CLI evaluation script.

### 1. Launching the Auto-Synthesizer (Streamlit UI)

To launch the secure offline web prototype, run:

streamlit run dpumis.py


* **Tab 1 (NYC Pre-trained Engine):** Generate context-conditioned synthetic trips based on pre-trained NYC TLC weights.

* **Tab 2 (Universal Auto-Synthesizer):** Upload your own proprietary CSV to dynamically train a private GAN and download a synthetic clone.

### 2. Running the Machine Learning Efficacy Test (TSTR)

To mathematically validate the utility of your synthetic data (Train on Synthetic, Test on Real), ensure your real and synthetic CSVs are in the root directory and run:

python tstr_evaluator.py


This script will output the $R^2$ scores, Mean Absolute Error (MAE), and generate a comparison bar chart.

## 📂 Repository Structure
```
PUMIS/
│
├── dpumis.py                 # Main Streamlit App & ContextGAN Architectures
├── tstr_evaluator.py         # Standalone ML Efficacy Evaluation Script
├── pumis_generator.pth       # Pre-trained Generator Weights (NYC TLC Model)
├── scaler_params.pth         # Normalization State Dictionary
├── original_nyc_data.csv     # Sample Ground Truth Data (Optional)
├── README.md                 # Project Documentation
└── requirements.txt          # Python Dependencies
```
## 👥 Authors

This project was developed by researchers at the Department of Information Technology, **Info Institute of Engineering, Coimbatore, India.**

* **Madeswaran K** (Assistant Professor & Project Guide)

* **Sivashankaran R** (Student Developer/Researcher)

* **Dharchana A** (Student Developer/Researcher)

* **Akash S** (Student Developer/Researcher)

* **Ashwin E** (Student Developer/Researcher)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

*Built for the future of Smart Cities and Secure Data Democratization.*
