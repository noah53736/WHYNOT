import streamlit as st
import os
import io
import requests
import time
from pydub import AudioSegment
import shutil
import pandas as pd

# === Configuration FFMPEG / FFPROBE ===
# Force pydub à utiliser le ffmpeg/ffprobe installé via packages.txt
AudioSegment.converter = shutil.which("ffmpeg")
AudioSegment.ffprobe = shutil.which("ffprobe")

# === Fonction de Transcription Nova (DeepGram) ===
def transcribe_nova_one_shot(
    file_bytes: bytes,
    dg_api_key: str,      # la clé Nova (NOVA / NOVA2 / NOVA3)
    language: str = "fr",
    model_name: str = "nova-2",
    punctuate: bool = True,
    numerals: bool = True
) -> str:
    """
    Envoie les données audio à Deepgram (Nova) et retourne la transcription.
    Convertit au préalable l'audio en PCM mono 16kHz.
    """
    try:
        # Conversion en 16kHz mono WAV
        temp_in = "temp_nova_in.wav"
        seg = AudioSegment.from_file(io.BytesIO(file_bytes))
        seg_16k = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        seg_16k.export(temp_in, format="wav")

        # Préparation des paramètres de la requête
        params = []
        params.append(f"language={language}")
        params.append(f"model={model_name}")
        if punctuate:
            params.append("punctuate=true")
        if numerals:
            params.append("numerals=true")

        qs = "?" + "&".join(params)
        url = f"https://api.deepgram.com/v1/listen{qs}"

        # Envoi de la requête à Deepgram
        with open(temp_in, "rb") as f:
            payload = f.read()

        headers = {
            "Authorization": f"Token {dg_api_key}",
            "Content-Type": "audio/wav"
        }
        response = requests.post(url, headers=headers, data=payload)

        # Nettoyage du fichier temporaire
        os.remove(temp_in)

        if response.status_code == 200:
            result = response.json()
            transcript = (
                result.get("results", {})
                      .get("channels", [{}])[0]
                      .get("alternatives", [{}])[0]
                      .get("transcript", "")
            )
            return transcript
        else:
            st.error(f"[Nova] Erreur {response.status_code} : {response.text}")
            return ""
    except Exception as e:
        st.error(f"Erreur pendant la transcription : {e}")
        return ""

# === Configuration de la Page Streamlit ===
st.set_page_config(page_title="Deepgram Transcription App", layout="wide")
st.title("Deepgram Transcription App")

# === Lecture des Clés API depuis les Secrets Streamlit ===
dg_key = st.secrets.get("NOVA", "")
if not dg_key:
    st.error("La clé API 'NOVA' n'est pas configurée. Ajoute-la dans les secrets Streamlit.")
    st.stop()

# === Options de Transcription ===
st.sidebar.header("Options Deepgram")
DG_MODELS = ["nova-2", "whisper-large"]  # Modèles Deepgram disponibles
chosen_model = st.sidebar.selectbox("Choisissez un modèle", DG_MODELS, index=0)
DEFAULT_LANGUAGE = "fr"
LANGUAGE_MAP = {
    "fr": "fr",
    "french": "fr",
    "f": "fr",
    "en": "en-US",
    "english": "en-US",
    "e": "en-US"
}
input_language = st.sidebar.text_input("Code langue (ex : 'fr' ou 'en')", value=DEFAULT_LANGUAGE)

def normalize_language(input_lang: str) -> str:
    """
    Normalise la langue saisie (indépendant de la casse ou des variantes).
    Retourne 'fr' pour français et 'en-US' pour anglais.
    """
    input_lang = input_lang.strip().lower()
    return LANGUAGE_MAP.get(input_lang, DEFAULT_LANGUAGE)  # Langue par défaut si la saisie est invalide.

language = normalize_language(input_language)

st.write(f"**Modèle choisi** : {chosen_model}")
st.write(f"**Langue choisie** : {language}")

# === Téléchargement ou Enregistrement Audio ===
st.write("## Téléchargez un fichier audio ou enregistrez depuis le microphone")
input_choice = st.radio("Choisissez une méthode d'entrée :", ["Télécharger un fichier", "Microphone"])

audio_data = None
if input_choice == "Télécharger un fichier":
    uploaded_file = st.file_uploader(
        "Téléchargez un fichier audio (formats acceptés : mp3, wav, m4a, ogg, webm)",
        type=["mp3", "wav", "m4a", "ogg", "webm"]
    )
    if uploaded_file:
        audio_data = uploaded_file.read()
        st.audio(uploaded_file)
elif input_choice == "Microphone":
    mic_input = st.audio_input("Enregistrez un audio via le micro")
    if mic_input:
        audio_data = mic_input.read()
        st.audio(mic_input)

# === Transcription ===
if audio_data and st.button("Transcrire"):
    try:
        st.write("Transcription en cours...")
        start_time = time.time()

        # Transcription avec Deepgram
        transcription = transcribe_nova_one_shot(
            file_bytes=audio_data,
            dg_api_key=dg_key,
            model_name=chosen_model,
            language=language
        )

        # Affichage des Résultats
        elapsed_time = time.time() - start_time
        if transcription:
            st.success(f"Transcription terminée en {elapsed_time:.2f} secondes")
            st.subheader("Résultat de la transcription")
            st.write(transcription)

            # Bouton pour télécharger la transcription
            st.download_button(
                "Télécharger la transcription",
                data=transcription,
                file_name="transcription.txt",
                mime="text/plain"
            )
        else:
            st.warning("Aucune transcription retournée. Vérifiez les logs ci-dessus.")
    except Exception as e:
        st.error(f"Erreur lors du traitement : {e}")
