import streamlit as st
import os
import io
import requests
import time
from pydub import AudioSegment
import shutil

# Optionnel : définir explicitement où se trouvent ffmpeg / ffprobe
# (Si la config de Streamlit Cloud ne trouve pas ffprobe par défaut)
AudioSegment.converter = shutil.which("ffmpeg")
AudioSegment.ffprobe   = shutil.which("ffprobe")

DG_MODELS = ["nova-2", "whisper-large"]  # Modèles Deepgram disponibles
DEFAULT_LANGUAGE = "fr"
LANGUAGE_MAP = {
    "fr": "fr",
    "french": "fr",
    "f": "fr",
    "en": "en-US",
    "english": "en-US",
    "e": "en-US"
}

st.set_page_config(page_title="Deepgram Transcription App", layout="wide")

def normalize_language(input_lang: str) -> str:
    """
    Normalise la langue saisie (en 'fr' ou 'en-US').
    Retourne 'fr' si rien n'est trouvé.
    """
    input_lang = input_lang.strip().lower()
    return LANGUAGE_MAP.get(input_lang, DEFAULT_LANGUAGE)

def transcribe_deepgram(file_bytes: bytes, api_key: str, model_name: str, language: str) -> str:
    """
    Envoie les données audio à Deepgram et retourne la transcription.
    Convertit au préalable l'audio en PCM mono 16kHz.
    """
    try:
        temp_in = "temp_dg_in.wav"

        # Conversion en 16kHz mono WAV
        seg = AudioSegment.from_file(io.BytesIO(file_bytes))
        seg_16k = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        seg_16k.export(temp_in, format="wav")

        params = f"?model={model_name}&language={language}&punctuate=true&numerals=true"
        url = f"https://api.deepgram.com/v1/listen{params}"

        with open(temp_in, "rb") as f:
            payload = f.read()

        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav"
        }
        response = requests.post(url, headers=headers, data=payload)

        # Nettoyage
        try:
            os.remove(temp_in)
        except:
            pass

        if response.status_code == 200:
            result = response.json()
            return (
                result.get("results", {})
                      .get("channels", [{}])[0]
                      .get("alternatives", [{}])[0]
                      .get("transcript", "")
            )
        else:
            st.error(f"[Deepgram Error] HTTP {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        st.error(f"Erreur pendant la transcription : {e}")
        return ""

def main():
    st.title("Deepgram Transcription App")

    # Lecture de la clé depuis les Secrets (Streamlit Cloud)
    dg_key = st.secrets.get("NOVA", "")
    if not dg_key:
        st.error("Clé 'NOVA' manquante dans les secrets Streamlit.")
        st.stop()

    st.sidebar.header("Options Deepgram")
    chosen_model = st.sidebar.selectbox("Modèle Deepgram", DG_MODELS, index=0)
    input_language = st.sidebar.text_input("Langue (fr / en)", value=DEFAULT_LANGUAGE)
    language = normalize_language(input_language)

    st.write(f"**Modèle choisi** : {chosen_model}")
    st.write(f"**Langue choisie** : {language}")

    st.write("## 1) Charger ou enregistrer un fichier audio")
    mode = st.radio("Choisissez une méthode :", ["Télécharger un fichier", "Microphone"])

    audio_data = None
    if mode == "Télécharger un fichier":
        uploaded_file = st.file_uploader(
            "Formats acceptés : mp3, wav, m4a, ogg, webm",
            type=["mp3", "wav", "m4a", "ogg", "webm"]
        )
        if uploaded_file:
            audio_data = uploaded_file.read()
            st.audio(uploaded_file)
    else:
        # Le micro Streamlit (depuis la v1.25)
        mic = st.audio_input("Enregistrements audio")
        if mic:
            audio_data = mic.read()
            st.audio(mic)

    if audio_data and st.button("Transcrire"):
        st.write("Transcription en cours...")
        start_time = time.time()

        transcription = transcribe_deepgram(
            file_bytes=audio_data,
            api_key=dg_key,
            model_name=chosen_model,
            language=language
        )

        duration = time.time() - start_time
        if transcription:
            st.success(f"Transcription OK en {duration:.2f} sec.")
            st.subheader("Résultat")
            st.write(transcription)

            st.download_button(
                label="Télécharger le texte",
                data=transcription,
                file_name="transcription.txt",
                mime="text/plain"
            )
        else:
            st.warning("La transcription est vide ou a échoué.")

if __name__ == "__main__":
    main()
