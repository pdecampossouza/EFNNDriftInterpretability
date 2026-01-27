
# 🧠 EFNN Drift & Interpretability — UC3M Seminar

Welcome to the **EFNN Drift & Interpretability** repository!  
This project supports a **hands-on seminar** delivered at **Universidad Carlos III de Madrid (UC3M)** on:

> **Evolving Fuzzy Neural Networks, Interpretability, and Concept Drift in Data Streams**

---

## 👨‍🏫 Instructor

**Prof. Dr. Paulo Vitor de Campos Souza**  
NOVA IMS – Universidade Nova de Lisboa  
📧 Contact: paulo.souza@novaims.unl.pt

---

## 🎯 Course Goals

By the end of this seminar, participants will be able to:

- 🔹 Understand **Fuzzy Neural Networks (FNNs)**  
- 🔹 Interpret fuzzy rules and membership functions  
- 🔹 Apply **Evolving Fuzzy Systems** to data streams  
- 🔹 Detect and analyze **Concept Drift**  
- 🔹 Compare evolving fuzzy models with online baselines  

All concepts are demonstrated through **interactive Jupyter notebooks**.

---

## 📂 Repository Structure

```
EFNNDriftInterpretability/
│
├── notebooks/
│   ├── Notebook1_FNN_Interpretability.ipynb
│   └── Notebook2_EvolvingFuzzySystems_Drift.ipynb
│
├── models/
│   └── models.py              # Fuzzy Neural Network model
│
├── experiments/
│   └── calculate.py           # Interpretability metrics
│
├── README.md                  # This file
└── requirements.txt           # Optional dependency list
```

---

## 🧪 Notebook Overview

### 📘 Notebook 1 — Fuzzy Neural Networks & Interpretability
- Fuzzification layers (Gaussian MFs)
- Rule generation and explosion
- Pseudo-inverse learning
- Interpretability metrics:
  - Consistency
  - Similarity
  - Distinguishability
  - e-Completeness
- Visual explanation of fuzzy rules

### 📕 Notebook 2 — Evolving Systems & Concept Drift
- What is a data stream?
- Types of concept drift:
  - Sudden
  - Gradual
  - Incremental
  - Recurring
- Prequential (online) evaluation
- Drift detection with **ADWIN**
- Comparison:
  - ENFS_Uni0 (Evolving Fuzzy Classifier)
  - River online baselines
- Visual drift markers and rolling accuracy

---

## ⚙️ Installation Guide (Quick Start)

### 1️⃣ Create a virtual environment (recommended)

**Windows**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Upgrade core tools
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3️⃣ Install dependencies
```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn tqdm river evolvingfuzzysystems jupyter
```

### 4️⃣ Launch Jupyter
```bash
jupyter notebook
```

---

## 🧠 Key Concepts Illustrated

- 🧩 Interpretability ≠ Black box  
- 🔁 Learning without retraining  
- 📈 Stability vs Adaptation  
- 🔍 Rules as knowledge units  
- 🚨 Drift-aware decision making  

---

## 📊 Evaluation Methodology

- **Prequential learning** (predict → learn)
- **Rolling accuracy**
- **ADWIN drift detection**
- Rule growth and pruning over time

---

## 📌 Notes

- Some numerical warnings (overflow, RLS instability) may appear — this is **expected** in adaptive systems and does not affect learning.
- The notebooks are designed to be **didactic**, not optimized for large-scale deployment.

---

## 📚 References

- Alves, K. S. T. R. *Evolvingfuzzysystems: A Python Library*. Zenodo, 2025.  
  🔗 https://doi.org/10.5281/zenodo.15748291

- P. V. C. Souza et al. *Evolving Fuzzy Neural Networks for Interpretable Learning*

---

## 🤝 Acknowledgements

Special thanks to:
- **Universidad Carlos III de Madrid (UC3M)**
- **NOVA IMS**
- **Kaike Alves** for the evolvingfuzzysystems library

---

## ⭐ How to Cite

If you use this material in academic work, please cite the repository:

```
Souza, P. V. C. (2026).
EFNN Drift & Interpretability.
GitHub repository.
https://github.com/pdecampossouza/EFNNDriftInterpretability
```

---

🚀 **Enjoy exploring interpretable evolving fuzzy systems!**
