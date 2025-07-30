# RF-ML-Defense

Machine Learning Pipeline for Detecting and Defending Against Adversarial Attacks in RF Signal Classification.

## Overview

This project focuses on building a machine learning pipeline to classify RF signals and develop defenses against adversarial attacks, specifically in the context of the RadioML2016.10a dataset.

The main components of the project include:
- Loading and preprocessing real and synthetic RF datasets
- Training baseline and adversarially robust classifiers
- Generating adversarial examples
- Evaluating model robustness and defense mechanisms

---

## Project Structure

```bash
RF-ML-Defense/
├── data/
│   ├── raw/
│   │   └── RadioML2016.10a/
│   │       └── RML2016.10a_dict.pkl
│   ├── processed/
│   │   ├── real_data.npz
│   │   └── synthetic_data.npz
├── models/
│   ├── trained_model.pth
│   └── adversarial_model.pth
├── notebooks/
│   └── EDA.ipynb
├── scripts/
│   ├── fetch_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── generate_adversarial.py
│   ├── evaluate_model.py
│   └── defenses/
│       └── apply_defense.py
├── utils/
│   └── helpers.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/RF-ML-Defense.git
   cd RF-ML-Defense
   ```

2. **Install dependencies**  
   It's recommended to use a virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**  
   Use the Kaggle CLI:
   ```bash
   kaggle datasets download -d nolasthitnotomorrow/radioml2016-deepsigcom -p ./data/raw --unzip
   ```
   Ensure the file structure is:
   ```
   data/raw/RadioML2016.10a/RML2016.10a_dict.pkl
   ```

---

## How to Run (in order)

In PyCharm terminal or any CLI:

1. **Fetch synthetic data (if not already present)**  
   ```bash
   python scripts/fetch_data.py
   ```

2. **Preprocess the real and synthetic datasets**  
   ```bash
   python scripts/preprocess.py
   ```

3. **Train the baseline model**  
   ```bash
   python scripts/train_model.py
   ```

4. **Generate adversarial examples**  
   ```bash
   python scripts/generate_adversarial.py
   ```

5. **Evaluate model on clean and adversarial examples**  
   ```bash
   python scripts/evaluate_model.py
   ```

6. **Apply defenses and re-evaluate**  
   ```bash
   python scripts/defenses/apply_defense.py
   ```

---

## Requirements

All requirements are listed in `requirements.txt`. Example packages:
- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `kaggle`

---

## Notes

- Synthetic data may be fetched or generated in `fetch_data.py` (if internet access to the relevant dataset is available).
- Preprocessing saves `.npz` files in `data/processed/`.
- Model weights are saved under `models/`.

---

## Contact

For questions or issues, please open an issue or contact Prathum Arikeri.

---

## Disclaimer

This is a preliminary skeleton. **Many scripts may contain errors or be incomplete.** Use at your own discretion and feel free to contribute.
