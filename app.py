# app.py

import streamlit as st
import pandas as pd
import os
import io
import time
import random
import string
from datetime import datetime
import json
import matplotlib.pyplot as plt
import subprocess  # Ajout pour la fonction accelerate_ffmpeg

from pydub import AudioSegment, silence

import nova_api  # Assure-toi que `nova_api.py` est dans le même répertoire

###############################################################################
# CONSTANTES
###############################################################################
HISTORY_FILE = "historique.csv"
CREDITS_FILE = "credits.json"

# Paramètres "silence"
MIN_SIL_MS = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
CROSSFADE_MS = 50

###############################################################################
# UTILITAIRES
###############################################################################
def init_history():
    if not os.path.exists(HISTORY_FILE):
        df_init = pd.DataFrame(columns=[
            "Alias", "Méthode", "Modèle", "Durée", "Temps", "Coût",
            "Transcription", "Date", "Audio Binaire"
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
                           min_sil_len=MIN_SIL_MS,
                           silence_thresh=SIL_THRESH_DB,
                           keep_sil_ms=KEEP_SIL_MS,
                           crossfade_ms=CROSSFADE_MS):
    segs = silence.split_on_silence(
        audio_seg,
        min_sil_len=min_sil_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_sil_ms
    )
    if not segs:
        return audio_seg
    combined = segs[0]
    for s in segs[1:]:
        combined = combined.append(s, crossfade=crossfade_ms)
    return combined

###############################################################################
# LOAD CREDITS
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
# GRAPH CRÉDITS
###############################################################################
def plot_credits(credits):
    total_credit = sum(credits.values())
    remaining_credit = sum([credit for credit in credits.values()])

    labels = ['Crédit Total', 'Crédit Restant']
    sizes = [total_credit, remaining_credit]
    colors = ['#ff9999','#66b3ff']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

###############################################################################
# MAIN STREAMLIT APP
###############################################################################
def main():
    st.set_page_config(page_title="NBL Audio", layout="wide")
    st.title("NBL Audio – Transcription Audio")

    # Initialiser / Charger l'historique
    if "hist_df" not in st.session_state:
        df = load_history()
        st.session_state["hist_df"] = df
    hist_df = st.session_state["hist_df"]

    # Charger les crédits
    credits = load_credits()

    # === Options de Transcription ===
    st.sidebar.header("Options Transcription")

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

    # Filtrer les clés API avec crédit suffisant
    selectable_keys = []
    key_number_mapping = {}  # Mapping de label à clé réelle
    for i, key_id in enumerate(key_ids):
        remaining = credits.get(key_id, 0)
        if remaining > 0:
            label = f"API Key {i+1}"
            selectable_keys.append(label)
            key_number_mapping[label] = key_id

    if not selectable_keys:
        st.sidebar.error("Toutes les clés API sont épuisées.")
        st.stop()

    # Afficher le crédit total et restant
    total_credit = sum(credits.values())
    remaining_credit = sum([credit for credit in credits.values()])
    st.sidebar.subheader("Crédits")
    progress = remaining_credit / total_credit if total_credit > 0 else 0
    st.sidebar.progress(progress)
    st.sidebar.write(f"**Total Crédit**: ${total_credit:.2f}")
    st.sidebar.write(f"**Crédit Restant**: ${remaining_credit:.2f}")
    plot_credits(credits)

    # === Transformations ===
    st.sidebar.write("---")
    st.sidebar.header("Transformations")
    remove_sil = st.sidebar.checkbox("Supprimer silences (douce)", False)
    speed_factor = st.sidebar.slider("Accélération (time-stretch)", 0.5, 4.0, 1.0, 0.1)

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

            # Charger les crédits
            credits = load_credits()

            # Liste des clés API ordonnées
            ordered_keys = sorted(key_ids, key=lambda x: int(x.replace("NOVA", "")))

            # Extraire les clés API dans l'ordre
            ordered_api_keys = [st.secrets[key] for key in ordered_keys]

            # Transcription avec Failover et segmentation
            transcription, used_key, cost = nova_api.transcribe_nova_one_shot(
                file_bytes=audio_data,
                api_keys=ordered_api_keys,
                language="fr",  # Modifier selon tes besoins
                model_name="whisper-large",  # Modifier selon tes besoins
                punctuate=True,
                numerals=True,
                credits=credits
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
                final_sec = len(final_aud) / 1000.0
                if final_sec < original_sec:
                    gain_sec = original_sec - final_sec
                gain_str = human_time(gain_sec)

                # Affichage des Informations
                st.write(f"Durée finale : {human_time(final_sec)} (gagné {gain_str}) | "
                         f"Temps effectif: {human_time(elapsed_time)} | Coût=${cost:.2f}")

                # Identifier quelle API Key a été utilisée
                if used_key in key_ids:
                    api_key_number = key_ids.index(used_key) + 1
                    st.write(f"**Clé API utilisée** : API Key {api_key_number}")
                else:
                    st.write(f"**Clé API utilisée** : {used_key}")

                # Enregistrer dans l'historique
                alias = generate_alias(6)
                audio_buf = audio_data  # Directement utiliser les bytes

                new_row = {
                    "Alias": alias,
                    "Méthode": "Nova (Deepgram)",
                    "Modèle": "whisper-large",
                    "Durée": f"{human_time(original_sec)}",
                    "Temps": human_time(elapsed_time),
                    "Coût": f"${cost:.2f}",
                    "Transcription": transcription,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": audio_buf
                }

                # Utiliser pd.concat au lieu de append
                new_df = pd.DataFrame([new_row])
                hist_df = pd.concat([hist_df, new_df], ignore_index=True)
                st.session_state["hist_df"] = hist_df
                save_history(hist_df)

                st.info(f"Historique mis à jour. Alias={alias}")

                # Rafraîchir les crédits affichés
                plot_credits(credits)

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
