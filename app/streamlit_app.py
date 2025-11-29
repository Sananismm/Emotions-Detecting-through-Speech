import os
import tempfile

import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import sounddevice as sd   # <-- NEW

# ------------------ CONFIG ------------------
SR = 16000
N_MFCC = 13
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

st.set_page_config(page_title="Speech Emotion Detection", layout="wide")

st.title("🎭 Emotion Detection from Speech")
st.write("MFCC-based DSP features + SVM trained on RAVDESS.")

# Debug info in sidebar so we know paths are OK
st.sidebar.header("Debug info")
st.sidebar.write("Working dir:", os.getcwd())
st.sidebar.write("Model dir:", MODEL_DIR)
try:
    st.sidebar.write("Files in model dir:", os.listdir(MODEL_DIR))
except Exception as e:
    st.sidebar.write("Error listing models:", e)

# ------------------ MODEL LOADING ------------------
@st.cache(allow_output_mutation=True)
def load_model_and_scaler():
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    model_path  = os.path.join(MODEL_DIR, "svm_ravdess.joblib")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Expected files not found in {MODEL_DIR}\n"
            f"scaler.joblib exists? {os.path.exists(scaler_path)}\n"
            f"svm_ravdess.joblib exists? {os.path.exists(model_path)}"
        )

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)
    return scaler, model


try:
    scaler, svm_clf = load_model_and_scaler()
    st.success("Model and scaler loaded successfully ✅")
except Exception as e:
    st.error("Could not load model/scaler.")
    st.exception(e)
    st.stop()   # stop here so user sees the error


# ------------------ DSP FEATURES ------------------

def extract_features(y, sr=SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    features = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1),
        zcr.mean(axis=1), zcr.std(axis=1),
        centroid.mean(axis=1), centroid.std(axis=1),
        bandwidth.mean(axis=1), bandwidth.std(axis=1),
    ])
    return features


def preprocess_array(y, sr=SR):
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    y_norm = librosa.util.normalize(y_trimmed)
    return y_norm


def predict_emotion_from_signal(signal, sr=SR):
    y_proc = preprocess_array(signal, sr=sr)
    feats = extract_features(y_proc, sr=sr)
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    label = svm_clf.predict(feats_scaled)[0]
    proba = svm_clf.predict_proba(feats_scaled)[0]
    return label, proba, y_proc


# ------------------ PLOTTING HELPERS ------------------

def plot_waveform(y, sr=SR, title="Waveform"):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


def plot_spectrogram(y, sr=SR, title="Spectrogram"):
    D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    DB = librosa.amplitude_to_db(D, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(DB, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)


def plot_probabilities(proba, classes, predicted_label):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(classes, proba)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(f"Emotion Probabilities (Predicted: {predicted_label})")
    st.pyplot(fig)


# ------------------ RECORDING HELPER ------------------

def record_audio(duration_sec=5, sr=SR):
    """Record audio from default microphone."""
    st.info(f"Recording for {duration_sec} seconds… Speak now 🎙️")
    audio = sd.rec(int(duration_sec * sr),
                   samplerate=sr,
                   channels=1,
                   dtype="float32")
    sd.wait()
    st.success("Recording complete.")
    return audio.flatten()


# ------------------ MAIN UI: MODE SELECTOR ------------------

mode = st.sidebar.radio(
    "Input mode:",
    ("📁 Upload WAV File", "🎙️ Record from Microphone")
)

duration = st.sidebar.slider("Recording duration (seconds)", 2, 8, 5)

# ---------- MODE 1: UPLOAD ----------
if mode == "📁 Upload WAV File":
    st.subheader("📁 Upload a speech WAV file")

    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

    if uploaded_file is not None:
        try:
            # Save uploaded file to a temp file so librosa can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load audio at SR=16000
            y_file, _ = librosa.load(tmp_path, sr=SR, mono=True)

            label, proba, y_proc = predict_emotion_from_signal(y_file, sr=SR)

            st.markdown(f"## Predicted Emotion: **{label}**")

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Processed waveform")
                plot_waveform(y_proc, sr=SR, title="Processed Waveform")
            with col2:
                st.caption("Spectrogram")
                plot_spectrogram(y_proc, sr=SR, title="Magnitude Spectrogram")

            classes = svm_clf.classes_
            plot_probabilities(proba, classes, label)

        except Exception as e:
            st.error("Error while processing the uploaded file.")
            st.exception(e)

# ---------- MODE 2: RECORD ----------
else:
    st.subheader("🎙️ Live Recording")

    if st.button("Start Recording"):
        try:
            y_live = record_audio(duration_sec=duration, sr=SR)

            label, proba, y_proc = predict_emotion_from_signal(y_live, sr=SR)

            st.markdown(f"## Predicted Emotion: **{label}**")

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Processed waveform")
                plot_waveform(y_proc, sr=SR, title="Processed Live Waveform")
            with col2:
                st.caption("Spectrogram")
                plot_spectrogram(y_proc, sr=SR, title="Live Magnitude Spectrogram")

            classes = svm_clf.classes_
            plot_probabilities(proba, classes, label)

        except Exception as e:
            st.error("Error during recording or prediction.")
            st.exception(e)
