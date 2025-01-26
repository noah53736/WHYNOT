import streamlit as st
import pandas as pd
import os
import io
import time
import random
import string
import subprocess
from datetime import datetime
import requests

from pydub import AudioSegment, silence

import nova_api  # Assure-toi que `nova_api.py` est dans le même répertoire

###############################################################################
# CONSTANTES
###############################################################################
HISTORY_FILE = "historique.csv"

# Paramètres "silence"
MIN_SIL_MS = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
CROSSFADE_MS = 50

# Coûts (ajuster selon ta tarification réelle)
NOVA_COST_PER_MINUTE = 0.007

###############################################################################
# HISTORIQUE
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

    # === Options de Transcription ===
    st.sidebar.header("Options Transcription")

    # Mapping des clés API à leurs modèles respectifs
    # Ajuste ce dictionnaire selon tes clés et modèles
    api_model_mapping = {
        "NOVA": "nova-2",
        "NOVA2": "whisper-large",
        "NOVA3": "whisper-large",  # Ajoute d'autres clés si nécessaire
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
            with open("temp_original.wav", "wb") as foo:
                foo.write(audio_data)

            aud = AudioSegment.from_file("temp_original.wav")
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

            # **Vérification de la Clé API**
            st.write(f"Longueur de la clé API lue : {len(dg_key)} caractères")

            # Transcription avec Deepgram
            transcription = nova_api.transcribe_nova_one_shot(
                file_bytes=audio_data,
                dg_api_key=dg_key,
                model_name=chosen_model,
                language=language,
                punctuate=True,
                numerals=True
            )

            # Affichage des Résultats
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

                # Calcul du coût
                mn = original_sec / 60.0
                cost_val = round(mn * NOVA_COST_PER_MINUTE, 4)
                cost_str = f"{cost_val} €"

                # Temps effectif
                total_proc = elapsed_time
                total_str = human_time(total_proc)

                # Gains en temps
                gain_sec = 0
                final_sec = len(final_aud) / 1000.0
                if final_sec < original_sec:
                    gain_sec = original_sec - final_sec
                gain_str = human_time(gain_sec)

                st.write(f"Durée finale : {human_time(final_sec)} (gagné {gain_str}) | "
                         f"Temps effectif: {total_str} | Coût={cost_str}")

                # Enregistrer dans l'historique
                alias = generate_alias(6)
                audio_buf = audio_data  # Directement utiliser les bytes

                new_row = {
                    "Alias": alias,
                    "Méthode": "Nova (Deepgram)",
                    "Modèle": chosen_model,
                    "Durée": f"{human_time(original_sec)}",
                    "Temps": total_str,
                    "Coût": cost_str,
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
