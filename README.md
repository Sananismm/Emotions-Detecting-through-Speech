# 🎤 Emotion Detection Through Speech

A machine learning project that classifies human emotions using **speech audio signals**, combining feature extraction, signal processing, and deep learning.

---

## 🚀 Overview

This project aims to automatically detect emotions from raw audio recordings using MFCC-based feature extraction and ML models. It is designed for:

* ML beginners exploring audio classification
* Researchers working with speech datasets (e.g., RAVDESS)
* Developers implementing emotion-aware applications

**Current Status:** Feature extraction, preprocessing pipeline, and initial model training completed.

---

## 🧠 Features

* MFCC-based audio feature extraction
* Automated preprocessing pipeline
* Emotion classification from speech
* Support for datasets like **RAVDESS**
* Modular `utils/` code design
* Jupyter Notebook for experiments and visualizations

---

## 📁 Project Structure

```plaintext
emotion_speech_project/
 ┣ utils/
 ┃ ┗ features.py
 ┣ notebooks/
 ┃ ┗ main_notebook.ipynb
 ┣ data/ (ignored from Git)
 ┣ models/
 ┣ README.md
 ┣ .gitignore
```

---

## 🛠️ Tech Stack

* **Python**
* **Librosa** – Audio loading & MFCC extraction
* **NumPy, Pandas** – Data handling
* **Matplotlib** – Visualizations
* **Scikit-learn / TensorFlow / PyTorch** – Model training

---

## 🔧 Installation

```bash
git clone <repo-url>
cd emotion_speech_project
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the notebook:

```bash
jupyter notebook notebooks/main_notebook.ipynb
```

Or process audio files:

```bash
python utils/features.py
```

---

## 📊 Results / Outputs

* MFCC visualizations
* Confusion matrix of model performance
* Accuracy and F1-score metrics

(Add screenshots or graphs once finalized.)

---

## 🧪 Testing

```bash
pytest
```

Or manually run feature extraction on sample WAV files.

---

## 🤝 Contributing

Pull requests, issues, and suggestions welcome!

---

## 📜 License

MIT / Apache / GPL — whichever you choose.

---

## 👨‍💻 Author

**Muhammad Sanan Khan**

