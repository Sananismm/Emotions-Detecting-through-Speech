import os
import tempfile

import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False


import tensorflow as tf

# ------------------ CONFIG ------------------
SR = 16000
N_MFCC = 13
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

st.set_page_config(page_title="Speech Emotion Detection", layout="wide")

st.title("🎭 Emotion Detection from Speech")
st.write("MFCC-based DSP features + SVM trained on RAVDESS.")
if model_choice == "CNN (Mel-Spectrogram)":
    st.success("🧠 Active Model: CNN (Mel-Spectrogram)")
else:
    st.info("📐 Active Model: SVM (MFCC Features)")


# ------------------ SIDEBAR CONTROLS ------------------

st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose model:",
    ("SVM (MFCC)", "CNN (Mel-Spectrogram)")
)


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


@st.cache_resource
def load_cnn_model():
    cnn_model = tf.keras.models.load_model("models/cnn_emotion_model_83.h5")
    cnn_label_encoder = joblib.load("models/cnn_label_encoder.pkl")
    return cnn_model, cnn_label_encoder

cnn_model, cnn_label_encoder = load_cnn_model()


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

def audio_to_melspec_for_cnn(y, sr=16000,
                             n_mels=128, n_fft=1024,
                             hop_length=256, max_frames=128):
    """
    Convert raw audio array into normalized Mel-spectrogram
    shaped for CNN inference.
    """
    # Trim + normalize
    y, _ = librosa.effects.trim(y, top_db=25)
    y = librosa.util.normalize(y)

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Fix time dimension
    n_frames = mel_db.shape[1]

    if n_frames < max_frames:
        pad_width = max_frames - n_frames
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=mel_db.min()
        )
    else:
        start = (n_frames - max_frames) // 2
        mel_db = mel_db[:, start:start + max_frames]

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # CNN expects (1, H, W, 1)
    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    return mel_db

def predict_emotion_cnn(y):
    mel_input = audio_to_melspec_for_cnn(y)
    probs = cnn_model.predict(mel_input)[0]
    idx = np.argmax(probs)
    emotion = cnn_label_encoder.inverse_transform([idx])[0]
    return emotion, probs

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

def plot_mel_spectrogram(y, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax
    )
    ax.set_title("Mel-Spectrogram (CNN Input)")
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

if HAS_SOUNDDEVICE:
    mode = st.sidebar.selectbox(
        "Select Mode",
        ("📁 Upload WAV File", "🎙️ Record Live Audio")
    )
else:
    mode = "Upload WAV File"
    st.sidebar.warning(
        "sounddevice module not available. "
        "Live recording disabled."
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

            if model_choice == "SVM (MFCC)":
                label, proba, y_proc = predict_emotion_from_signal(y_file, sr=SR)
                classes = svm_clf.classes_
            else:  # CNN
                label, proba = predict_emotion_cnn(y_file)
                y_proc = preprocess_array(y_file, sr=SR)
                classes = cnn_label_encoder.classes_

            st.markdown(f"## Predicted Emotion: **{label}**")

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Processed waveform")
                plot_waveform(y_proc, sr=SR, title="Processed Waveform")
            with col2:
                if model_choice == "CNN (Mel-Spectrogram)":
                    st.caption("Mel-Spectrogram (CNN Input)")
                    plot_mel_spectrogram(y_proc, sr=SR)
                else:
                    st.caption("Spectrogram")
                    plot_spectrogram(y_proc, sr=SR, title="Magnitude Spectrogram")

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

            if model_choice == "SVM (MFCC)":
                label, proba, y_proc = predict_emotion_from_signal(y_live, sr=SR)
                classes = svm_clf.classes_
            else:  # CNN
                label, proba = predict_emotion_cnn(y_live)
                y_proc = preprocess_array(y_live, sr=SR)
                classes = cnn_label_encoder.classes_

            st.markdown(f"## Predicted Emotion: **{label}**")

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Processed waveform")
                plot_waveform(y_proc, sr=SR, title="Processed Live Waveform")
            with col2:
                st.caption("Spectrogram")
                plot_spectrogram(y_proc, sr=SR, title="Live Magnitude Spectrogram")

            plot_probabilities(proba, classes, label)

        except Exception as e:
            st.error("Error during recording or prediction.")
            st.exception(e)



# streamlit run app/streamlit_app.py
