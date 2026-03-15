import streamlit as st
import librosa
import numpy as np
import torch
import soundfile as sf
import plotly.graph_objects as go
import plotly.express as px
from model import AudioTransformer

st.set_page_config(page_title="AI Audio Repair Studio", layout="wide")

st.title("🎧 AI Audio Repair Studio")
st.markdown("Restore missing audio using **AI Audio Inpainting**")

# Sidebar controls
st.sidebar.header("⚙ Controls")

start_frame = st.sidebar.slider("Missing Region Start",0,200,40)
end_frame = st.sidebar.slider("Missing Region End",start_frame,300,60)

uploaded_file = st.file_uploader("Upload Audio", type=["wav","mp3"])

if uploaded_file:

    with open("input_audio.wav","wb") as f:
        f.write(uploaded_file.read())

    audio, sr = librosa.load("input_audio.wav", sr=16000)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Audio")
        st.audio("input_audio.wav")

    # waveform visualization
    time = np.arange(len(audio))/sr

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time,y=audio,mode='lines'))
    fig.update_layout(title="Waveform",xaxis_title="Time",yaxis_title="Amplitude")

    st.plotly_chart(fig,use_container_width=True)

    # spectrogram
    spec = librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=80)
    spec_db = librosa.power_to_db(spec)

    fig2 = px.imshow(spec_db,aspect='auto',origin='lower',
                     title="Mel Spectrogram")

    st.plotly_chart(fig2,use_container_width=True)

    if st.button("🚀 Run AI Reconstruction"):

        spec_db = spec_db.T

        mask = np.ones(len(spec_db))
        mask[start_frame:end_frame] = 0

        corrupted = spec_db.copy()
        corrupted[mask==0] = 0

        model = AudioTransformer()
        model.eval()

        x = torch.tensor(corrupted).float().unsqueeze(0)

        with torch.no_grad():
            pred = model(x)

        pred = pred.squeeze().numpy()

        reconstructed = spec_db.copy()
        reconstructed[mask==0] = pred[mask==0]

        reconstructed = reconstructed.T

        audio_out = librosa.feature.inverse.mel_to_audio(
            librosa.db_to_power(reconstructed),
            sr=sr
        )

        sf.write("reconstructed.wav",audio_out,sr)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.audio("input_audio.wav")

        with col2:
            st.subheader("Reconstructed")
            st.audio("reconstructed.wav")

        with open("reconstructed.wav","rb") as f:
            st.download_button(
                "⬇ Download Reconstructed Audio",
                f,
                file_name="reconstructed.wav"
            )

        st.success("AI Reconstruction Complete")