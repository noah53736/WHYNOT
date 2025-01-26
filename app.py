import streamlit as st
import os
import io
from pydub import AudioSegment
import requests
import time

# ============================
# Constants & Setup
# ============================
DG_MODELS = ["nova-2", "whisper-large"]  # Modèles disponibles
DEFAULT_LANGUAGE = "fr"  # Langue par défaut
LANGUAGE_MAP = {
    "fr": "fr",
    "french": "fr",
    "f": "fr",
    "en": "en-US",
    "english": "en-US",
    "e": "en-US"
}
DG_COST_PER_MINUTE = 0.007  # Coût par minute de transcription, ajustable si nécessaire

st.set_page_config(page_title="Deepgram Transcription App", layout="wide")


# ============================
# Helper Functions
# ============================
def normalize_language(input_lang: str) -> str:
    """
    Normalise la langue saisie (indépendant de la casse ou des variantes).
    Retourne 'fr' pour français et 'en-US' pour anglais.
    """
    input_lang = input_lang.strip().lower()
    return LANGUAGE_MAP.get(input_lang, DEFAULT_LANGUAGE)  # Langue par défaut si la saisie est invalide.


def transcribe_deepgram(file_bytes: bytes, api_key: str, model_name: str, language: str) -> str:
    """
    Envoie les données audio à Deepgram et retourne la transcription.
    """
    try:
        # Convertir en 16kHz mono PCM pour Deepgram
        tmp_in = "temp_dg_in.wav"
        seg = AudioSegment.from_file(io.BytesIO(file_bytes))
        seg_16k = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        seg_16k.export(tmp_in, format="wav")

        # Requête API Deepgram
        params = f"?model={model_name}&language={language}&punctuate=true&numerals=true"
        url = f"https://api.deepgram.com/v1/listen{params}"
        with open(tmp_in, "rb") as f:
            payload = f.read()

        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav"
        }
        response = requests.post(url, headers=headers, data=payload)

        if response.status_code == 200:
            result = response.json()
            return result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        else:
            st.error(f"[Deepgram Error] HTTP {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        st.error(f"Erreur pendant la transcription : {e}")
        return ""


# ============================
# Main App
# ============================
def main():
    st.title("Deepgram Transcription App")

    # Lecture des clés API depuis les secrets Streamlit
    dg_key = st.secrets.get("NOVA", "")
    if not dg_key:
        st.error("La clé API 'NOVA' n'est pas configurée. Ajoutez-la dans les secrets Streamlit.")
        st.stop()

    # Choix des options
    st.sidebar.header("Options Deepgram")
    chosen_model = st.sidebar.selectbox("Choisissez un modèle", DG_MODELS, index=0)
    input_language = st.sidebar.text_input("Code langue (ex : 'fr' ou 'en')", value=DEFAULT_LANGUAGE)
    language = normalize_language(input_language)

    st.write(f"**Modèle choisi** : {chosen_model}")
    st.write(f"**Langue choisie** : {language}")

    # Chargement ou enregistrement d'audio
    st.write("## Téléchargez un fichier audio ou enregistrez depuis le microphone")
    input_choice = st.radio("Choisissez une méthode d'entrée :", ["Télécharger un fichier", "Microphone"])

    audio_data = None
    if input_choice == "Télécharger un fichier":
        uploaded_file = st.file_uploader("Téléchargez un fichier audio (formats acceptés : mp3, wav, m4a, ogg, webm)", type=["mp3", "wav", "m4a", "ogg", "webm"])
        if uploaded_file:
            audio_data = uploaded_file.read()
            st.audio(uploaded_file)
    elif input_choice == "Microphone":
        mic_input = st.audio_input("Enregistrez un audio via le micro")
        if mic_input:
            audio_data = mic_input.read()
            st.audio(mic_input)

    # Transcrire si l'audio est présent
    if audio_data and st.button("Transcrire"):
        try:
            st.write("Transcription en cours...")
            start_time = time.time()

            # Transcription avec Deepgram
            transcription = transcribe_deepgram(audio_data, dg_key, chosen_model, language)

            # Résultats
            elapsed_time = time.time() - start_time
            st.success(f"Transcription terminée en {elapsed_time:.2f} secondes")
            st.subheader("Résultat de la transcription")
            st.write(transcription)

            # Bouton pour télécharger la transcription
            st.download_button("Télécharger la transcription", data=transcription, file_name="transcription.txt", mime="text/plain")
        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")


# Lancer l'application
if __name__ == "__main__":
    main()
