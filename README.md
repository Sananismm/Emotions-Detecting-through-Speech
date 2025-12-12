

# 🎭 Speech Emotion Recognition System

**Signals & Systems Project | Real-Time Emotion Detection from Speech**

---

## 📌 Overview

This project implements a **Speech Emotion Recognition (SER)** system that classifies human emotions from spoken audio signals.
It combines **classical signal processing techniques** with **machine learning** and **deep learning**, and is deployed as an **interactive Streamlit web application**.

The system supports:

* 📁 Emotion detection from uploaded WAV files
* 🎙️ Real-time emotion detection from live microphone input (local execution)
* 🔀 Model switching between:

  * **SVM with MFCC-based DSP features**
  * **CNN with Mel-Spectrogram inputs**

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset for training and evaluation.

---

## 🧠 Models Implemented

### 1️⃣ SVM (Classical DSP + Machine Learning)

* Features:

  * MFCCs
  * Delta MFCCs
  * Zero-Crossing Rate
  * Spectral Centroid
  * Spectral Bandwidth
* Feature normalization using `StandardScaler`
* Classifier: Support Vector Machine (SVM)
* Strengths:

  * Interpretable
  * Lightweight
  * Strong signal-processing foundation

---

### 2️⃣ CNN (Deep Learning)

* Input representation: **Log Mel-Spectrograms**
* Architecture:

  * 2D Convolutional layers
  * Max pooling
  * Dense layers with softmax output
* Achieved accuracy: **~83%**
* Strengths:

  * Learns time–frequency patterns automatically
  * Higher accuracy than classical approach

---

## 📊 Signal Processing Pipeline

### SVM Pipeline

```
Raw Audio
 → Silence Trimming
 → Normalization
 → MFCC + Spectral Feature Extraction
 → Feature Scaling
 → SVM Classification
```

### CNN Pipeline

```
Raw Audio
 → Silence Trimming
 → Normalization
 → Mel-Spectrogram
 → Log Scaling
 → CNN Inference
```

---

## 🖥️ Web Application (Streamlit)

The Streamlit app provides:

* Model selection (SVM / CNN)
* Upload-based emotion detection
* Live microphone recording (local execution only)
* Visualization of:

  * Waveform
  * Spectrogram / Mel-Spectrogram
  * Emotion probability distribution

---

## 📂 Project Structure

```text
emotion_speech_project/
│
├── app/
│   └── streamlit_app.py        # Main Streamlit application
│
├── models/
│   ├── svm_ravdess.joblib      # Trained SVM model
│   ├── scaler.joblib           # Feature scaler
│   ├── cnn_emotion_model_83.h5 # Trained CNN model
│   └── cnn_label_encoder.pkl   # CNN label encoder
│
├── notebooks/
│   └── 01_train_SVM.ipynb      # Feature extraction & SVM training
│
├── utils/
│   └── feature utilities      # Signal processing helpers
│
├── data/
│   └── RAVDESS/                # Dataset (not included in repo)
│
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

See `requirements.txt`.

Key dependencies:

* Python 3.9+
* Streamlit
* Librosa
* NumPy
* Scikit-learn
* TensorFlow
* Matplotlib
* SoundDevice (local recording only)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## 🎙️ Live Recording Support

* Live microphone recording works **only on local machines**
* Disabled automatically on cloud deployments
* Upload-based inference works everywhere

---

## 📈 Dataset

* **RAVDESS Dataset**
* Emotions include:

  * Neutral
  * Calm
  * Happy
  * Sad
  * Angry
  * Fearful
  * Disgust
  * Surprised

Dataset is not included due to licensing.

---

## 🎓 Academic Context

This project was developed as part of a **Signals & Systems course**, with emphasis on:

* Time-domain and frequency-domain analysis
* Feature extraction from audio signals
* Practical application of DSP concepts
* Comparison of classical ML vs deep learning

---

## 🚀 Future Improvements

* Browser-based microphone recording
* Data augmentation
* Transfer learning (pretrained audio CNNs)
* Real-time emotion timeline visualization
* Multi-language emotion recognition

---

## 👨‍💻 Author

**Muhammad Sanan Khan**
Electrical Engineering
Speech & Signal Processing Project

---

## 📜 License

This project is for **educational and research purposes only**.


Just say the word 👍
