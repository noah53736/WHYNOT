import streamlit as st
import os
import io
import requests
import time
from pydub import AudioSegment, silence
import shutil
import pandas as pd
import random
import string
import subprocess

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
    Envoie les données audio à Deepgram (Nova) et retourne la transcription (str).
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

# === Gestion de l'Historique ===
HISTORY_FILE = "historique.csv"

def init_history():
    if not os.path.exists(HISTORY_FILE):
        df_init = pd.DataFrame(columns=[
            "Alias","Méthode","Modèle","Durée","Temps","Coût",
            "Transcription","Date","Audio Binaire"
        ])
        df_init.to_csv(HISTORY_FILE, index=False)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        init_history()
    return pd.read_csv(HISTORY_FILE)

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)

def generate_alias(length=5):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

def human_time(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    elif sec < 3600:
        m, s = divmod(sec, 60)
        return f"{m}m{s}s"
    else:
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        return f"{h}h{m}m{s}s"

# === Accélération (Time-Stretch) ===
def accelerate_ffmpeg(audio_seg: AudioSegment, factor: float) -> AudioSegment:
    """
    Accélère ou ralentit le signal via ffmpeg (atempo).
    factor=2 => x2 plus rapide
    factor=0.5 => 2x plus lent
    """
    if abs(factor - 1.0) < 1e-2:
        return audio_seg  # Pas de changement

    tmp_in = "temp_acc_in.wav"
    tmp_out = "temp_acc_out.wav"
    audio_seg.export(tmp_in, format="wav")

    remain = factor
    filters = []
    # Pour atempo, la valeur doit être entre 0.5 et 2.0
    while remain > 2.0:
        filters.append("atempo=2.0")
        remain /= 2.0
    while remain < 0.5:
        filters.append("atempo=0.5")
        remain /= 0.5
    filters.append(f"atempo={remain}")
    f_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y", "-i", tmp_in,
        "-filter:a", f_str,
        tmp_out
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new_seg = AudioSegment.from_file(tmp_out, format="wav")
    try:
        os.remove(tmp_in)
        os.remove(tmp_out)
    except:
        pass
    return new_seg

# === Suppression de Silences (Douce) ===
MIN_SIL_MS = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
CROSSFADE_MS = 50

def remove_silences_smooth(audio_seg: AudioSegment,
                           min_sil_len=MIN_SIL_MS,
                           silence_thresh=SIL_THRESH_DB,
                           keep_sil_ms=KEEP_SIL_MS,
                           crossfade_ms=CROSSFADE_MS):
    """
    Découpe la piste sur les silences, puis recolle le tout
    en fondant les transitions (crossfade).
    """
    segs = silence.split_on_silence(
        audio_seg,
        min_sil_len=min_sil_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_sil_ms
    )
    if not segs:
        return audio_seg  # Rien trouvé, ou piste vide

    combined = segs[0]
    for s in segs[1:]:
        combined = combined.append(s, crossfade=crossfade_ms)
    return combined

# === Main Streamlit App ===
def main():
    st.set_page_config(page_title="Deepgram Transcription App", layout="wide")
    st.title("Deepgram Transcription App")

    # Initialiser / Charger l'historique
    if "hist_df" not in st.session_state:
        df = load_history()
        st.session_state["hist_df"] = df
    hist_df = st.session_state["hist_df"]

    # === Options de Transcription ===
    st.sidebar.header("Options Deepgram")

    # Mapping des clés API à leurs modèles respectifs
    # Ajuste ce dictionnaire selon tes clés et modèles
    api_model_mapping = {
        "NOVA": "nova-2",
        "NOVA2": "whisper-large",
        # Ajoute d'autres clés si nécessaire
    }

    # Récupérer toutes les clés API définies dans les secrets
    available_keys = [key for key in st.secrets.keys() if key.startswith("NOVA")]

    if not available_keys:
        st.sidebar.error("Aucune clé API DeepGram (NOVA) trouvée dans les secrets.")
        st.stop()

    chosen_key = st.sidebar.selectbox("Choisissez une clé API DeepGram", available_keys)

    # Déterminer le modèle en fonction de la clé choisie
    chosen_model = api_model_mapping.get(chosen_key, "nova-2")  # Modèle par défaut si non trouvé

    # Lire la clé API depuis les secrets
    dg_key = st.secrets[chosen_key]

    # Déterminer la langue
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

    st.write(f"**Clé API utilisée** : {chosen_key}")
    st.write(f"**Modèle utilisé** : {chosen_model}")
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

            # **Vérification de la Clé API**
            st.write(f"Longueur de la clé API lue : {len(dg_key)} caractères")

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

                # Enregistrer dans l'historique
                alias = generate_alias()
                method = input_choice
                duration = human_time(elapsed_time)
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                # Coût estimé (ajuster selon ta tarification DeepGram)
                # Par exemple, supposons 0.01€ par minute
                cost = round(elapsed_time / 60 * 0.01, 4)

                new_entry = {
                    "Alias": alias,
                    "Méthode": method,
                    "Modèle": chosen_model,
                    "Durée": duration,
                    "Temps": elapsed_time,
                    "Coût": cost,
                    "Transcription": transcription,
                    "Date": current_time,
                    "Audio Binaire": audio_data
                }

                # Ajouter la nouvelle entrée à l'historique
                st.session_state["hist_df"] = st.session_state["hist_df"].append(new_entry, ignore_index=True)
                save_history(st.session_state["hist_df"])

            else:
                st.warning("Aucune transcription retournée. Vérifie les logs ci-dessus.")
        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")

    # === Gestion de l'Historique ===
    st.sidebar.write("---")
    st.sidebar.header("Historique")

    disp_cols = ["Alias", "Méthode", "Modèle", "Durée", "Temps", "Coût", "Transcription", "Date"]
    if not hist_df.empty:
        show_cols = [c for c in disp_cols if c in hist_df.columns]
        st.sidebar.dataframe(hist_df[show_cols][::-1], use_container_width=True)
    else:
        st.sidebar.info("Historique vide.")

    st.sidebar.write("### Derniers Audios (3)")
    if not hist_df.empty:
        last_auds = hist_df[::-1].head(3)
        for idx, row in last_auds.iterrows():
            ab = row.get("Audio Binaire", None)
            if isinstance(ab, bytes):
                st.sidebar.write(f"**{row.get('Alias', '?')}** – {row.get('Date', '?')}")
                st.sidebar.audio(ab, format="audio/wav")

if __name__ == "__main__":
    main()
