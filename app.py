# app.py

import streamlit as st
import os
import io
import random
import string
from datetime import datetime
from pydub import AudioSegment
import nova_api
import time

######################################
#  INITIALISATION, UTILITAIRES ETC.  #
######################################
def init_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "dg_key_index" not in st.session_state:
        st.session_state["dg_key_index"] = 0

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

def copy_to_clipboard(text):
    script_html = f"""
    <script>
        function copyText(txt) {{
            navigator.clipboard.writeText(txt).then(function() {{
                alert('Transcription copiée dans le presse-papiers!');
            }}, function(err) {{
                alert('Échec de la copie : ' + err);
            }});
        }}
    </script>
    <button onclick="copyText(`{text}`)" style="margin:5px;">Copier</button>
    """
    st.components.v1.html(script_html)

def display_history():
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    h = st.session_state["history"]
    if h:
        table = []
        for en in h:
            table.append({
                "Alias/Nom": en["Alias/Nom"],
                "Méthode": en["Méthode"],
                "Modèle": en["Modèle"],
                "Durée": en["Durée"],
                "Temps": en["Temps"],
                "Coût": en["Coût"],
                "Date": en["Date"]
            })
        st.sidebar.table(table[::-1])
    else:
        st.sidebar.info("Historique vide.")

def get_api_keys():
    """Récupérer toutes les clés API DeepGram (prefix= 'NOVA')."""
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def pick_key(api_keys):
    """On prend la clé courante, sans rotation multiple."""
    if not api_keys:
        return None
    idx = st.session_state["dg_key_index"]
    if idx >= len(api_keys):
        st.session_state["dg_key_index"] = 0
        idx = 0
    return api_keys[idx]

######################################
#           LOGIQUE PRINCIPALE       #
######################################
def main():
    st.set_page_config(page_title="NBL Audio", layout="wide")
    st.title("NBL Audio")

    init_state()
    history = st.session_state["history"]
    dg_keys = get_api_keys()

    # 1. Vérification
    if not dg_keys:
        st.error("Aucune clé API configurée.")
        st.stop()

    # 2. Entrée Audio
    st.subheader("Entrée Audio")
    audio_data_list = []
    file_names = []
    input_type = st.radio("Source :", ["Fichier (Upload)","Micro (Enregistrement)"], index=0)
    if input_type=="Fichier (Upload)":
        upf = st.file_uploader("Fichiers audio", type=["mp3","wav","m4a","ogg","webm"], accept_multiple_files=True)
        if upf:
            for f in upf:
                audio_data_list.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        nmics = st.number_input("Nombre de micros", min_value=1, max_value=4, value=1)
        for i in range(nmics):
            mic = st.audio_input(f"Micro {i+1}")
            if mic:
                audio_data_list.append(mic.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic, format=mic.type)

    st.write("---")

    # 3. Options
    st.subheader("Options de Transcription")
    colA, colB = st.columns([1,1])
    with colA:
        model_main = st.selectbox("Modèle Principal", ["Nova 2","Whisper Large"])
    with colB:
        lang_main = "fr"
        if model_main=="Whisper Large":
            lang_main = st.selectbox("Langue Whisper", ["fr","en"])
    double_trans = st.checkbox("Double Transcription (Nova2 d'abord, puis Whisper)")

    # 4. Bouton Transcrire
    if st.button("Transcrire") and audio_data_list:
        for idx, raw_data in enumerate(audio_data_list):
            st.write(f"### Fichier {idx+1}")
            f_name = file_names[idx] if idx<len(file_names) else f"File_{idx+1}"
            seg = AudioSegment.from_file(io.BytesIO(raw_data))
            orig_sec = len(seg)/1000.0
            st.write(f"Durée = {human_time(orig_sec)}")

            # On sauvegarde le WAV
            temp_wav = f"temp_input_{idx}.wav"
            seg.export(temp_wav, format="wav")

            if double_trans:
                # 1) On fait Nova 2
                st.write(f"**{f_name} => Nova 2**")
                start_nova = time.time()
                key_nova = pick_key(dg_keys)
                txt_nova = nova_api.transcribe_audio(temp_wav, key_nova, "fr", "nova-2")
                end_nova = time.time()

                # On affiche le panel gauche / droit
                leftC, rightC = st.columns(2)
                with leftC:
                    st.subheader("NOVA 2")
                    st.text_area(f"{f_name} - Nova2", txt_nova, height=120)
                    copy_to_clipboard(txt_nova)

                eNova = {
                    "Alias/Nom": f"{f_name}_Nova2",
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": human_time(orig_sec),
                    "Temps": f"{(end_nova-start_nova):.1f}s",
                    "Coût": "?",
                    "Transcription": txt_nova,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(eNova)

                # 2) Whisper
                st.write(f"**{f_name} => Whisper Large**")
                start_whisp = time.time()
                key_whisp = pick_key(dg_keys)
                txt_whisp = nova_api.transcribe_audio(temp_wav, key_whisp, lang_main, "whisper-large")
                end_whisp = time.time()

                with rightC:
                    st.subheader("WHISPER LARGE")
                    st.text_area(f"{f_name} - Whisper", txt_whisp, height=120)
                    copy_to_clipboard(txt_whisp)

                eWhisp = {
                    "Alias/Nom": f"{f_name}_Whisper",
                    "Méthode": "Whisper Large",
                    "Modèle": "whisper-large",
                    "Durée": human_time(orig_sec),
                    "Temps": f"{(end_whisp-start_whisp):.1f}s",
                    "Coût": "?",
                    "Transcription": txt_whisp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(eWhisp)
            else:
                # Simple
                if model_main=="Nova 2":
                    chosen_model="nova-2"
                    chosen_lang="fr"
                else:
                    chosen_model="whisper-large"
                    chosen_lang=lang_main

                st.write(f"**{f_name} => {model_main}**")
                start_simp = time.time()
                d_key = pick_key(dg_keys)
                txt_single = nova_api.transcribe_audio(temp_wav, d_key, chosen_lang, chosen_model)
                end_simp = time.time()

                leftC, _ = st.columns([1,1])
                with leftC:
                    st.subheader(f"{model_main}")
                    st.text_area(f"{f_name} - Résultat", txt_single, height=120)
                    copy_to_clipboard(txt_single)

                eSingle = {
                    "Alias/Nom": f"{f_name}_{chosen_model}",
                    "Méthode": model_main,
                    "Modèle": chosen_model,
                    "Durée": human_time(orig_sec),
                    "Temps": f"{(end_simp-start_simp):.1f}s",
                    "Coût": "?",
                    "Transcription": txt_single,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(eSingle)

            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        display_history()
        st.write("---")
        st.write("### Aperçu Audio")
        for en in st.session_state["history"]:
            st.audio(bytes.fromhex(en["Audio Binaire"]), format="audio/wav")

def main_wrapper():
    try:
        main()
    except Exception as e:
        st.error(f"Erreur Principale : {e}")

if __name__=="__main__":
    main_wrapper()
