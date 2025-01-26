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
import subprocess  # Pour l'accélération audio

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

WHISPER_API_COST_PER_MINUTE = 0.006
LOCAL_COST_EUR_PER_HOUR = 4.0
NOVA_COST_PER_MINUTE = 0.007

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
# LOCAL WHISPER
###############################################################################
@st.cache_resource
def load_local_model(model_name: str):
    import whisper
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"[Local Whisper] Chargement du modèle {model_name} sur {dev}...")
    return whisper.load_model(model_name, device=dev)

def transcribe_local(file_path: str, model_name: str) -> str:
    model = load_local_model(model_name)
    result = model.transcribe(file_path)
    return result["text"]

###############################################################################
# OPENAI
###############################################################################
def transcribe_openai(file_path: str, openai_key: str) -> str:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {openai_key}"}
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"model": "whisper-1", "response_format": "text"}
        r = requests.post(url, headers=headers, files=files, data=data)
    if r.status_code == 200:
        return r.text
    else:
        st.error(f"[OpenAI] Erreur {r.status_code} : {r.text}")
        return ""

###############################################################################
def main():
    st.set_page_config(page_title="NBL Audio – 3 modes", layout="wide")
    st.title("NBL Audio – Local / OpenAI / Nova")

    if "hist_df" not in st.session_state:
        df = load_history()
        st.session_state["hist_df"] = df
    hist_df = st.session_state["hist_df"]

    # Choix du mode
    st.sidebar.header("Mode de Transcription")
    mode = st.sidebar.radio("Choix :", ["Local Whisper", "OpenAI Whisper", "Nova (Deepgram)"])

    # OpenAI key
    openai_key = os.getenv("OPENAI_APIKEY", "")  # Nom ajusté pour correspondre au requirements

    # Nova keys : "NOVA", "NOVA2", "NOVA3"
    # On propose un select
    nova_key_choice = st.sidebar.selectbox(
        "Clé Nova à utiliser",
        ["NOVA", "NOVA2", "NOVA3"]
    )
    chosen_nova_key = os.getenv(nova_key_choice, "")

    # Param local
    local_model = None
    if mode == "Local Whisper":
        local_model = st.sidebar.selectbox("Modèle local Whisper", 
            ["tiny", "base", "small", "medium", "large", "large-v2"], index=1)

    # Param Nova
    st.sidebar.write("---")
    lang_nova = "fr"
    model_nova = "nova-2"
    punct_nova = True
    numerals_nova = True
    if mode == "Nova (Deepgram)":
        st.sidebar.subheader("Deepgram Nova – Paramètres")
        lang_nova = st.sidebar.text_input("Langue (ex: 'fr', 'en-US')", "fr")
        # ex "nova-2","whisper-base","whisper-medium","whisper-large"
        model_nova = st.sidebar.selectbox("Modèle Deepgram", 
            ["nova-2", "whisper-base", "whisper-medium", "whisper-large"], index=0
        )
        punct_nova = st.sidebar.checkbox("Ponctuation (punctuate)", True)
        numerals_nova = st.sidebar.checkbox("Nombres => chiffres (numerals)", True)

    st.sidebar.write("---")
    st.sidebar.header("Transformations")
    remove_sil = st.sidebar.checkbox("Supprimer silences (douce)", False)
    speed_factor = st.sidebar.slider("Accélération (time-stretch)", 0.5, 4.0, 1.0, 0.1)
    remove_euh = st.sidebar.checkbox("Retirer 'euh' ?", False)

    st.sidebar.write("---")
    st.sidebar.header("Historique")
    # Colonnes => "Alias","Méthode","Modèle","Durée","Temps","Coût","Transcription","Date","Audio Binaire"
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
                st.sidebar.write(f"**{row.get('Alias','?')}** – {row.get('Date','?')}")
                st.sidebar.audio(ab, format="audio/wav")

    st.write("## Source Audio")
    audio_data = None
    input_choice = st.radio("Fichier ou Micro ?", ["Fichier", "Micro"])
    if input_choice == "Fichier":
        upf = st.file_uploader("Fichier audio (mp3, wav, m4a, ogg, webm)", 
                              type=["mp3", "wav", "m4a", "ogg", "webm"])
        if upf:
            if upf.size > 100 * 1024 * 1024:  # 100 MB
                st.error("Le fichier est trop volumineux. La taille maximale autorisée est de 100MB.")
            else:
                audio_data = upf.read()
                st.audio(upf)
    else:
        mic_in = st.audio_input("Enregistrement micro")
        if mic_in:
            audio_data = mic_in.read()
            st.audio(mic_in)

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

    if audio_data:
        if st.button("Transcrire maintenant"):
            start_t = time.time()
            with open("temp_input.wav", "wb") as f:
                final_aud.export(f, format="wav")

            final_txt = ""
            cost_str = ""
            final_sec = len(final_aud) / 1000.0

            if mode == "Local Whisper":
                if not local_model:
                    st.error("Aucun modèle local")
                    return
                txt = transcribe_local("temp_input.wav", local_model)
                final_txt = txt
                cost_str = "0.00 €"
                st.info("[Local] Coût=0€")

            elif mode == "OpenAI Whisper":
                if not openai_key:
                    st.error("Clé 'OPENAI_APIKEY' manquante.")
                    return
                txt = transcribe_openai("temp_input.wav", openai_key)
                final_txt = txt
                mn = final_sec / 60.0
                cost_val = round(mn * WHISPER_API_COST_PER_MINUTE, 3)
                cost_str = f"{cost_val} $"
                st.info(f"[OpenAI] ~{cost_str}")

            else:
                # Nova
                if not chosen_nova_key:
                    st.error(f"La clé '{nova_key_choice}' est vide ?")
                    return
                txt = nova_api.transcribe_nova_one_shot(
                    "temp_input.wav",
                    dg_api_key=chosen_nova_key,
                    language=lang_nova,
                    model_name=model_nova,
                    punctuate=punct_nova,
                    numerals=numerals_nova
                )
                final_txt = txt
                mn = final_sec / 60.0
                cost_val = round(mn * NOVA_COST_PER_MINUTE, 3)
                cost_str = f"{cost_val} $"
                st.info(f"[Nova] ~{cost_str}")

            # Remove 'euh'
            if remove_euh:
                new_txt, ccc = remove_euh_in_text(final_txt)
                final_txt = new_txt
                st.write(f"({ccc} 'euh' supprimés)")

            total_proc = time.time() - start_t
            total_str = human_time(total_proc)

            # Gains
            gain_sec = 0
            if final_sec < original_sec:
                gain_sec = original_sec - final_sec
            gain_str = human_time(gain_sec)

            st.success(f"Durée finale : {human_time(final_sec)} (gagné {gain_str}) | "
                       f"Temps effectif: {total_str} | Coût={cost_str}")

            st.write("## Résultat Final")
            st.text_area("Texte :", final_txt, height=200)
            st.download_button("Télécharger",
                data=final_txt.encode("utf-8"),
                file_name="transcription.txt"
            )

            # Mettre à jour historique
            alias = generate_alias(6)
            audio_buf = audio_data  # Directement utiliser les bytes

            used_model = ""
            if mode == "Local Whisper":
                used_model = local_model
            elif mode == "OpenAI Whisper":
                used_model = "whisper-1"
            else:
                used_model = model_nova

            new_row = {
                "Alias": alias,
                "Méthode": mode,
                "Modèle": used_model,
                "Durée": f"{human_time(original_sec)}",
                "Temps": total_str,
                "Coût": cost_str,
                "Transcription": final_txt,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Audio Binaire": audio_buf
            }
            hist_df = pd.concat([hist_df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state["hist_df"] = hist_df
            save_history(hist_df)

            st.info(f"Historique mis à jour. Alias={alias}")

    def main_wrapper():
        main()

    if __name__ == "__main__":
        main_wrapper()
    ```

---

## **4. Vérifications et Tests**

### **a. Vérifier la Présence et le Nom Correct de `runtime.txt`**

Assure-toi que le fichier `runtime.txt` est **nommé exactement** `runtime.txt` et qu'il se trouve **à la racine** de ton repository GitHub. Toute différence dans le nom ou l'emplacement empêchera Streamlit Cloud de le reconnaître.

### **b. Installer les Packages Localement (Optionnel mais Recommandé)**

Avant de pousser les modifications sur GitHub, il est recommandé de tester l'installation des dépendances localement avec Python 3.10 pour s'assurer qu'il n'y a pas de conflits.

**Action :**

1. **Créer un Environnement Virtuel avec Python 3.10 :**

   ```bash
   python3.10 -m venv env
   source env/bin/activate
