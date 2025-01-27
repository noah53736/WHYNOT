# app.py

import streamlit as st
import os
import io
import time
import random
import string
from datetime import datetime
import json
import matplotlib.pyplot as plt
import subprocess  # Pour l'accélération audio

from pydub import AudioSegment, silence

import nova_api  # Assure-toi que `nova_api.py` est dans le même répertoire

###############################################################################
# CONSTANTES
###############################################################################
HISTORY_FILE = "historique.json"
CREDITS_FILE = "credits.json"

# Paramètres "silence"
MIN_SILENCE_LEN = 700  # Correction de l'argument
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50

# Coût par seconde (à ajuster selon ton modèle de tarification DeepGram)
COST_PER_SEC = 0.007

###############################################################################
# UTILITAIRES
###############################################################################
def init_history():
    if not os.path.exists(HISTORY_FILE):
        history = []
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        init_history()
    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

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

###############################################################################
# ACCELERATION (TIME-STRETCH)
###############################################################################
def accelerate_ffmpeg(audio_seg: AudioSegment, factor: float) -> AudioSegment:
    if abs(factor - 1.0) < 1e-2:
        return audio_seg
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

###############################################################################
# SUPPRESSION SILENCES "CLASSIQUE"
###############################################################################
def remove_silences_classic(audio_seg: AudioSegment,
                           min_silence_len=MIN_SILENCE_LEN,
                           silence_thresh=SIL_THRESH_DB,
                           keep_silence=KEEP_SIL_MS):
    segs = silence.split_on_silence(
        audio_seg,
        min_silence_len=min_silence_len,  # Correction de l'argument
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    if not segs:
        return audio_seg
    combined = segs[0]
    for s in segs[1:]:
        combined = combined.append(s, crossfade=0)  # Pas de crossfade
    return combined

###############################################################################
# SELECTION AUTOMATIQUE DE LA CLÉ API
###############################################################################
def select_api_key(credits, key_ids, duration_sec, cost_per_sec=COST_PER_SEC):
    cost = duration_sec * cost_per_sec
    for key_id in key_ids:
        if credits.get(key_id, 0) >= cost:
            return key_id, cost
    return None, 0

###############################################################################
# MAIN STREAMLIT APP
###############################################################################
def main():
    st.set_page_config(page_title="NBL Audio – DeepGram", layout="wide")
    st.title("NBL Audio – DeepGram")

    # Initialiser / Charger l'historique
    if "history" not in st.session_state:
        history = load_history()
        st.session_state["history"] = history
    else:
        history = st.session_state["history"]

    # Charger les crédits
    credits = load_credits()

    # === Options de Transcription ===
    # Utiliser un conteneur unique pour mobile
    with st.sidebar:
        st.header("Options Transcription")

        # Sélection du modèle de transcription
        model_selection = st.selectbox(
            "Sélectionner le Modèle de Transcription",
            ["Nova 2", "Whisper Large"]
        )

        # Définir le nom du modèle basé sur la sélection
        model_mapping = {
            "Nova 2": "nova-2",
            "Whisper Large": "whisper-large"  # Assure-toi que ce modèle est disponible dans DeepGram
        }
        selected_model = model_mapping.get(model_selection, "nova-2")

        # Sélection de la langue
        language_selection = st.selectbox(
            "Sélectionner la Langue",
            ["fr", "en"]
        )

    # Charger les clés API depuis les secrets
    api_keys = []
    key_ids = []
    for key in st.secrets:
        if key.startswith("NOVA"):
            api_keys.append(st.secrets[key])
            key_ids.append(key)

    if not api_keys:
        st.sidebar.error("Aucune clé API DeepGram trouvée dans les secrets.")
        st.stop()

    # === Transformations ===
    with st.sidebar:
        st.write("---")
        st.header("Transformations")
        remove_sil = st.checkbox("Supprimer silences", False)
        speed_factor = st.slider("Accélération (time-stretch)", 0.5, 4.0, 1.0, 0.1)

    # === Téléchargement ou Enregistrement Audio ===
    st.write("## Source Audio")
    audio_data = None
    file_name = None  # Pour stocker le nom du fichier si renommé

    input_container = st.container()
    with input_container:
        input_choice = st.radio("Fichier ou Micro ?", ["Fichier", "Micro"], key="input_choice")
    
        if input_choice == "Fichier":
            uploaded_file = st.file_uploader("Fichier audio (mp3, wav, m4a, ogg, webm)", 
                                            type=["mp3", "wav", "m4a", "ogg", "webm"], key="uploaded_file")
            if uploaded_file:
                if uploaded_file.size > 100 * 1024 * 1024:  # 100 MB
                    st.error("Le fichier est trop volumineux. La taille maximale autorisée est de 100MB.")
                else:
                    audio_data = uploaded_file.read()
                    st.audio(uploaded_file, format=uploaded_file.type)
                    # Renommer le fichier
                    file_name = st.text_input("Renommer le fichier (optionnel)", key="rename_input")
        else:
            mic_input = st.audio_input("Enregistrement micro", key="mic_input")
            if mic_input:
                audio_data = mic_input.read()
                st.audio(mic_input, format=mic_input.type)

    # Boutons pour nettoyer les fichiers chargés et l'historique
    clean_container = st.container()
    with clean_container:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Clear Uploaded File"):
                audio_data = None
                st.experimental_rerun()
        with col2:
            if st.button("Clear History"):
                history = []
                st.session_state["history"] = history
                save_history(history)
                st.sidebar.success("Historique vidé.")

    # === Prétraitement Audio ===
    if audio_data:
        try:
            aud = AudioSegment.from_file(io.BytesIO(audio_data))
            original_sec = len(aud) / 1000.0
            st.write(f"Durée d'origine : {human_time(original_sec)}")

            final_aud = aud
            if remove_sil:
                final_aud = remove_silences_classic(final_aud)
            if abs(speed_factor - 1.0) > 1e-2:
                final_aud = accelerate_ffmpeg(final_aud, speed_factor)

            final_sec = len(final_aud) / 1000.0
            st.write(f"Durée finale après transformations : {human_time(final_sec)}")

            bufp = io.BytesIO()
            final_aud.export(bufp, format="wav")
            st.write("### Aperçu Audio transformé")
            st.audio(bufp.getvalue(), format="audio/wav")

            # Notification après gains de temps
            if final_sec < original_sec:
                gain_sec = original_sec - final_sec
                gain_str = human_time(gain_sec)
                st.success(f"Gagné {gain_str} grâce aux transformations audio.")

        except Exception as e:
            st.error(f"Erreur chargement/preproc : {e}")

    # === Transcription ===
    if audio_data and st.button("Transcrire"):
        try:
            st.write("Transcription en cours...")
            start_t = time.time()

            # Calculer la durée et le coût
            duration_sec = final_sec
            cost = duration_sec * COST_PER_SEC

            # Sélectionner une clé API disponible
            selected_key_id, actual_cost = select_api_key(credits, key_ids, duration_sec, COST_PER_SEC)
            if not selected_key_id:
                st.error("Toutes les clés API sont épuisées ou insuffisantes pour cette transcription.")
                st.stop()

            selected_api_key = st.secrets[selected_key_id]

            # Transcription via DeepGram Nova ou Whisper
            with open("temp_input.wav", "wb") as f:
                final_aud.export(f, format="wav")

            transcription = nova_api.transcribe_audio(
                file_path="temp_input.wav",
                dg_api_key=selected_api_key,
                language=language_selection,        # Langue sélectionnée
                model_name=selected_model,          # Basé sur la sélection utilisateur
                punctuate=True,
                numerals=True
            )

            elapsed_time = time.time() - start_t
            if transcription:
                st.success(f"Transcription terminée en {elapsed_time:.2f} secondes")
                st.subheader("Résultat de la transcription")
                
                # Afficher la transcription dans un champ de texte pour faciliter la copie
                st.text_area("Transcription", transcription, height=200)

                # Bouton pour télécharger la transcription
                st.download_button(
                    "Télécharger la transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

                # Notification pour copier la transcription
                st.info("Pour copier la transcription, sélectionnez le texte ci-dessus et utilisez CTRL+C.")

                # Calcul des gains en temps
                gain_sec = 0
                if final_sec < original_sec:
                    gain_sec = original_sec - final_sec
                gain_str = human_time(gain_sec)

                st.write(f"Durée finale : {human_time(final_sec)} (gagné {gain_str}) | "
                         f"Temps effectif: {human_time(elapsed_time)} | Coût=${actual_cost:.2f}")

                # Enregistrer dans l'historique
                alias = generate_alias(6) if not file_name else file_name
                entry = {
                    "Alias/Nom": alias,
                    "Méthode": f"{model_selection} (DeepGram)",
                    "Modèle": selected_model,
                    "Durée": human_time(original_sec),
                    "Temps": human_time(elapsed_time),
                    "Coût": f"${actual_cost:.2f}",
                    "Transcription": transcription,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": audio_data.hex()  # Stocker en hexadécimal
                }
                history.append(entry)
                st.session_state["history"] = history
                save_history(history)

                st.info(f"Historique mis à jour. Alias/Nom={alias}")

                # Mettre à jour les crédits
                credits[selected_key_id] -= actual_cost
                save_credits(credits)

                # Nettoyer le fichier temporaire
                if os.path.exists("temp_input.wav"):
                    os.remove("temp_input.wav")

        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")

    # === Gestion de l'Historique ===
    st.sidebar.write("---")
    st.sidebar.header("Historique")

    if history:
        # Afficher l'historique sous forme de tableau compact
        history_table = [
            {
                "Alias/Nom": entry["Alias/Nom"],
                "Méthode": entry["Méthode"],
                "Modèle": entry["Modèle"],
                "Durée": entry["Durée"],
                "Temps": entry["Temps"],
                "Coût": entry["Coût"],
                "Date": entry["Date"]
            }
            for entry in history
        ]
        st.sidebar.table(history_table[::-1])  # Afficher les plus récentes en haut

        # Afficher les aperçus audio
        st.sidebar.write("### Aperçus Audio")
        for entry in reversed(history[-3:]):  # Afficher les 3 dernières transcriptions
            st.sidebar.markdown(f"**{entry['Alias/Nom']}** – {entry['Date']}")
            audio_bytes = bytes.fromhex(entry["Audio Binaire"])
            st.sidebar.audio(audio_bytes, format="audio/wav")
    else:
        st.sidebar.info("Historique vide.")

###############################################################################
# FUNCTIONS FOR CREDITS
###############################################################################
def load_credits():
    if os.path.exists(CREDITS_FILE):
        with open(CREDITS_FILE, "r") as f:
            credits = json.load(f)
    else:
        # Initialiser les crédits si le fichier n'existe pas
        credits = {}
    return credits

def save_credits(credits):
    with open(CREDITS_FILE, "w") as f:
        json.dump(credits, f, indent=4)

###############################################################################
# FUNCTIONS FOR HISTORY
###############################################################################
def load_history():
    if not os.path.exists(HISTORY_FILE):
        init_history()
    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

###############################################################################
# MAIN WRAPPER
###############################################################################
def main_wrapper():
    main()

if __name__ == "__main__":
    main_wrapper()
