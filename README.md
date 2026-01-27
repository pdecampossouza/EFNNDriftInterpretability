# 📘 EFNN Drift & Interpretability — Course Notebooks

Welcome! This repository contains **didactic notebooks and scripts** used in the course / seminar on  
**Fuzzy Neural Networks, Evolving Fuzzy Systems, Concept Drift, and Interpretability**.

The structure below reflects the **current folder organization** of the project.

---

## 📂 Repository Structure

```
EFNNDriftInterpretability/
│
├── models/
│   ├── models.py                 # Core FNN / EFNN models
│   ├── evolving_nf_advanced.py   # Advanced evolving NF (used in drift demos)
│   └── operators.py              # Fuzzy operators
│
├── experiments/
│   ├── calculate.py              # Interpretability matrices (consistency, similarity, overlap)
│   ├── evaluation.py             # Evaluation helpers
│   ├── kg.py                     # Knowledge graph construction
│   ├── kgfuzzyrules.py           # Fuzzy rules → KG utilities
│   ├── plots.py                  # Plotting utilities
│   └── utils.py                  # Shared helpers
│
├── Notebook1_FNN_UC3M_EN_v1_1.ipynb
├── Notebook2_EFS_IA02_Interpretability_in_Evolution.ipynb
├── Notebook2_UC3M_EvolvingFuzzySystems_Drift_EN.ipynb
│
└── README.md
```

---

## 📓 Notebooks Overview

### 🧠 Notebook 1 — FNN & Interpretability
**`Notebook1_FNN_UC3M_EN_v1_1.ipynb`**

Focus:
- Fuzzy Neural Networks (FNN)
- Fuzzification and membership functions
- Rule extraction
- Interpretability matrices:
  - Consistency
  - Similarity
  - Distinguishability
  - ε‑Completeness
- Knowledge graphs from fuzzy rules

This notebook uses:
- `models/models.py`
- `experiments/calculate.py`
- `experiments/kg*.py`

---

### 🔄 Notebook 2 — Evolving Fuzzy Systems & Interpretability
**`Notebook2_EFS_IA02_Interpretability_in_Evolution.ipynb`**

Focus:
- Evolving Fuzzy Systems (eFS)
- Rule evolution over time
- Interpretability during learning
- Visualization of evolving rules

Uses the **evolvingfuzzysystems** library and local utilities.

---

### 🌊 Notebook 3 — Data Streams & Concept Drift
**`Notebook2_UC3M_EvolvingFuzzySystems_Drift_EN.ipynb`**

Focus:
- Data streams
- Concept drift (sudden, gradual, incremental, recurring)
- Prequential evaluation
- Drift detection (e.g. ADWIN)
- Comparison of multiple eFS models
- Rule growth vs accuracy trade‑off

Uses:
- `models/evolving_nf_advanced.py`
- `experiments/*`
- `river` (for synthetic drift streams)

---

## ⚙️ Requirements

### 🐍 Python
- **Python ≥ 3.10** (recommended: 3.11)

### 📦 Main Libraries
```
numpy
scipy
pandas
matplotlib
seaborn
scikit-learn
networkx
river
evolvingfuzzysystems
jupyterlab
```

Install everything with:

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn networkx river evolvingfuzzysystems jupyterlab
```

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/pdecampossouza/EFNNDriftInterpretability.git
cd EFNNDriftInterpretability
```

2. (Optional) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
```

3. Start Jupyter:
```bash
jupyter lab
```

4. Open the notebooks in order:
- Notebook 1 → fundamentals & interpretability
- Notebook 2 → evolving systems
- Notebook 3 → drift & streams

---

## 🎓 For Students

✔ All notebooks are **self‑contained**  
✔ Heavy experiments are **optional**  
✔ Focus on:
- Concepts
- Visualizations
- Interpretability insights

You do **not** need to understand all code details to follow the lecture.

---

## 📚 Citation

If you use the evolving fuzzy systems library, please cite:

> SA TELES ROCHA ALVES, K. (2025). *Evolvingfuzzysystems: a new Python library*. Zenodo.  
> https://doi.org/10.5281/zenodo.15748291

📄 Offline / Batch learning paper
```
@article{DECAMPOSSOUZA2021231,
title = {An evolving neuro-fuzzy system based on uni-nullneurons with advanced interpretability capabilities},
journal = {Neurocomputing},
volume = {451},
pages = {231-251},
year = {2021},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2021.04.065},
url = {https://www.sciencedirect.com/science/article/pii/S092523122100607X},
author = {Paulo Vitor {de Campos Souza} and Edwin Lughofer},
keywords = {Evolving neuro-fuzzy system (ENFS-Uni0), Uni-nullneurons, On-line interpretability of fuzzy rules, Degree of rule changes, Incremental feature importance levels, Indicator-based recursive weighted least squares (I-RWLS)},
}
```
🌊 Online / Evolving / Data stream paper
```
@article{DECAMPOSSOUZA2024121002,
title = {IFNN: Enhanced interpretability and optimization in FNN via Adam algorithm},
journal = {Information Sciences},
volume = {678},
pages = {121002},
year = {2024},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2024.121002},
url = {https://www.sciencedirect.com/science/article/pii/S0020025524009162},
author = {Paulo Vitor {de Campos Souza} and Mauro Dragoni},
keywords = {Fuzzy neural networks, Adam optimization, Interpretability of fuzzy rules, Fuzzy rules},
}

```
---

## 👨‍🏫 Instructor

**Prof. Dr. Paulo Vitor de Campos Souza**  
NOVA IMS – Universidade Nova de Lisboa  
📧 Contact: paulo.souza@novaims.unl.pt

Enjoy the notebooks and happy learning! 🚀
