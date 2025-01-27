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
MIN_SIL_MS = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
CROSSFADE_MS = 50

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

def remove_euh_in_text(text: str):
    words = text.split()
    cleaned = []
    cnt = 0
    for w in words:
        wlow = w.lower().strip(".,!?;:")
        if "euh" in wlow or "heu" in wlow:
            cnt += 1
        else:
            cleaned.append(w)
    return (" ".join(cleaned), cnt)

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
# SUPPRESSION SILENCES "douce"
###############################################################################
def remove_silences_smooth(audio_seg: AudioSegment,
                           min_silence_len=MIN_SIL_MS,
                           silence_thresh=SIL_THRESH_DB,
                           keep_silence=KEEP_SIL_MS,
                           crossfade_ms=CROSSFADE_MS):
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
        combined = combined.append(s, crossfade=crossfade_ms)
    return combined

###############################################################################
# GRAPH CRÉDITS
###############################################################################
def plot_credits(credits):
    initial_credit_per_key = 10.0  # Assumes chaque clé commence avec 10.0 crédits
    total_credit = len(credits) * initial_credit_per_key
    remaining_credit = sum(credits.values())
    used_credit = total_credit - remaining_credit

    labels = ['Crédit Utilisé', 'Crédit Restant']
    sizes = [used_credit, remaining_credit]
    colors = ['#ff9999','#66b3ff']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Assure que le graphique est un cercle.
    st.sidebar.pyplot(fig)

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
    st.sidebar.header("Options Transcription")

    # Sélection du modèle de transcription
    model_selection = st.sidebar.selectbox(
        "Sélectionner le Modèle de Transcription",
        ["Whisper Large", "Nova 2"]
    )

    # Définir le nom du modèle basé sur la sélection
    model_mapping = {
        "Whisper Large": "whisper-large",  # Assurez-vous que ce modèle est disponible dans DeepGram
        "Nova 2": "nova-2"
    }
    selected_model = model_mapping.get(model_selection, "nova-2")

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
    st.sidebar.write("---")
    st.sidebar.header("Transformations")
    remove_sil = st.sidebar.checkbox("Supprimer silences (douce)", False)
    speed_factor = st.sidebar.slider("Accélération (time-stretch)", 0.5, 4.0, 1.0, 0.1)
    remove_euh = st.sidebar.checkbox("Retirer 'euh' ?", False)

    # === Téléchargement ou Enregistrement Audio ===
    st.write("## Source Audio")
    audio_data = None
    input_choice = st.radio("Fichier ou Micro ?", ["Fichier", "Micro"])
    if input_choice == "Fichier":
        uploaded_file = st.file_uploader("Fichier audio (mp3, wav, m4a, ogg, webm)", 
                                        type=["mp3", "wav", "m4a", "ogg", "webm"])
        if uploaded_file:
            if uploaded_file.size > 100 * 1024 * 1024:  # 100 MB
                st.error("Le fichier est trop volumineux. La taille maximale autorisée est de 100MB.")
            else:
                audio_data = uploaded_file.read()
                st.audio(uploaded_file)
    else:
        mic_input = st.audio_input("Enregistrement micro")
        if mic_input:
            audio_data = mic_input.read()
            st.audio(mic_input)

    # === Prétraitement Audio ===
    if audio_data:
        try:
            aud = AudioSegment.from_file(io.BytesIO(audio_data))
            original_sec = len(aud) / 1000.0
            st.write(f"Durée d'origine : {human_time(original_sec)}")

            final_aud = aud
            if remove_sil:
                final_aud = remove_silences_smooth(final_aud)
            if abs(speed_factor - 1.0) > 1e-2:
                final_aud = accelerate_ffmpeg(final_aud, speed_factor)

            final_sec = len(final_aud) / 1000.0
            st.write(f"Durée finale après transformations : {human_time(final_sec)}")

            bufp = io.BytesIO()
            final_aud.export(bufp, format="wav")
            st.write("### Aperçu Audio transformé")
            st.audio(bufp.getvalue(), format="audio/wav")

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

            transcription = nova_api.transcribe_nova_one_shot(
                file_path="temp_input.wav",
                dg_api_key=selected_api_key,
                language="fr",        # Modifier selon tes besoins
                model_name=selected_model,  # Basé sur la sélection utilisateur
                punctuate=True,
                numerals=True
            )

            elapsed_time = time.time() - start_t
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

                # Calcul des gains en temps
                gain_sec = 0
                if final_sec < original_sec:
                    gain_sec = original_sec - final_sec
                gain_str = human_time(gain_sec)

                st.write(f"Durée finale : {human_time(final_sec)} (gagné {gain_str}) | "
                         f"Temps effectif: {human_time(elapsed_time)} | Coût=${actual_cost:.2f}")

                # Enregistrer dans l'historique
                alias = generate_alias(6)
                entry = {
                    "Alias": alias,
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

                st.info(f"Historique mis à jour. Alias={alias}")

                # Mettre à jour les crédits
                credits[selected_key_id] -= actual_cost
                save_credits(credits)

                # Rafraîchir les crédits affichés
                plot_credits(credits)

                # Nettoyer le fichier temporaire
                if os.path.exists("temp_input.wav"):
                    os.remove("temp_input.wav")

                # Optionnel: Retirer 'euh' si sélectionné
                if remove_euh:
                    new_txt, ccc = remove_euh_in_text(transcription)
                    transcription = new_txt
                    st.write(f"({ccc} 'euh' supprimés)")

            else:
                st.warning("Aucune transcription retournée. Vérifie les logs ci-dessus.")
        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")

    # === Gestion de l'Historique ===
    st.sidebar.write("---")
    st.sidebar.header("Historique")

    if history:
        for entry in reversed(history[-10:]):  # Afficher les 10 dernières transcriptions
            st.sidebar.markdown(f"**{entry['Alias']}** – {entry['Date']}")
            st.sidebar.text(entry['Transcription'][:100] + '...')  # Afficher un aperçu
    else:
        st.sidebar.info("Historique vide.")

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

def plot_credits(credits):
    initial_credit_per_key = 10.0  # Assumes chaque clé commence avec 10.0 crédits
    total_credit = len(credits) * initial_credit_per_key
    remaining_credit = sum(credits.values())
    used_credit = total_credit - remaining_credit

    labels = ['Crédit Utilisé', 'Crédit Restant']
    sizes = [used_credit, remaining_credit]
    colors = ['#ff9999','#66b3ff']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Assure que le graphique est un cercle.
    st.sidebar.pyplot(fig)

def main_wrapper():
    main()

if __name__ == "__main__":
    main_wrapper()
