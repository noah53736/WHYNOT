# app.py

import streamlit as st
import os
import io
import random
import string
from datetime import datetime
from pydub import AudioSegment
import nova_api

############################################
#         INITIALISATION ET UTILS          #
############################################

def init_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    # index de clé unique, on ne tourne pas
    # on tentera la clé suivante en cas d'échec
    if "key_index" not in st.session_state:
        st.session_state["key_index"] = 0

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

def display_history(history):
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    if history:
        table = []
        for en in history:
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

############################################
#          GESTION UNIQUE DES CLÉS         #
############################################

def get_api_keys():
    """ Récupérer toutes les clés API depuis st.secrets. """
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def try_transcription(file_path, keys, language="fr", model="nova-2"):
    """
    Tente de transcrire en utilisant la clé st.session_state["key_index"].
    Si échec => on passe à la clé suivante, etc.
    Retourne la transcription finale (string) ou "" en cas d'échec complet.
    """
    index = st.session_state["key_index"]
    n_keys = len(keys)
    attempt = 0
    while attempt < n_keys:
        if index >= n_keys:
            index = 0
        dg_key = keys[index]
        index += 1
        attempt += 1
        txt = nova_api.transcribe_audio(
            file_path,
            dg_key,
            language,
            model
        )
        if txt != "":
            # ok -> on retient ce nouvel index
            st.session_state["key_index"] = index - 1
            return txt
    # echec complet
    return ""

############################################
#         LOGIQUE PRINCIPALE APP           #
############################################

def main():
    st.set_page_config(page_title="NBL Audio", layout="wide")
    st.title("NBL Audio")

    init_state()
    history = st.session_state["history"]
    keys = get_api_keys()
    if not keys:
        st.error("Aucune clé API configurée.")
        st.stop()

    # 1) Entrée Audio
    st.subheader("Entrée Audio")
    audio_data = []
    file_names = []

    input_type = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"])
    if input_type=="Fichier (Upload)":
        upf = st.file_uploader("Uploadez vos fichiers audio", 
            type=["mp3","wav","m4a","ogg","webm"], 
            accept_multiple_files=True
        )
        if upf:
            for f in upf:
                audio_data.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        max_mics = 4
        nb_mics = st.number_input("Nombre de micros :", min_value=1, max_value=max_mics, value=1)
        for i in range(nb_mics):
            mic = st.audio_input(f"Micro {i+1}")
            if mic:
                audio_data.append(mic.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic, format=mic.type)

    st.write("---")

    # 2) Options
    st.subheader("Options de Transcription")
    colA, colB = st.columns([1,1])
    with colA:
        model_selection = st.selectbox("Modèle Principal :", ["Nova 2", "Whisper Large"])
    with colB:
        if model_selection=="Whisper Large":
            language_selection = st.selectbox("Langue :", ["fr","en"])
        else:
            language_selection = "fr"

    double_trans = st.checkbox("Double Transcription (Nova 2 puis Whisper Large)")

    # 3) Bouton Transcrire
    if st.button("Transcrire") and audio_data:
        st.info("Début de la transcription...")
        for idx, raw_data in enumerate(audio_data):
            seg = AudioSegment.from_file(io.BytesIO(raw_data))
            f_name = file_names[idx] if idx<len(file_names) else f"Fichier_{idx+1}"
            orig_sec = len(seg)/1000.0

            st.write(f"**Fichier {idx+1}: {f_name}** / Durée: {human_time(orig_sec)}")

            # Un fichier .wav provisoire
            temp_filename = f"temp_work_{idx}.wav"
            seg.export(temp_filename, format="wav")

            if double_trans:
                # 1) Nova 2
                txt_nova = try_transcription(temp_filename, keys, "fr", "nova-2")
                st.subheader(f"[Nova 2] {f_name}")
                st.text_area("Résultat Nova 2", txt_nova, height=150)
                copy_to_clipboard(txt_nova)

                # On ajoute un "entry" dans l'historique
                e1 = {
                    "Alias/Nom": f"{f_name}_Nova2",
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": human_time(orig_sec),
                    "Temps": human_time(orig_sec),
                    "Coût": "?",
                    "Transcription": txt_nova,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(e1)

                # 2) Whisper Large
                txt_whisper = try_transcription(temp_filename, keys, language_selection, "whisper-large")
                st.subheader(f"[Whisper Large] {f_name}")
                st.text_area("Résultat Whisper", txt_whisper, height=150)
                copy_to_clipboard(txt_whisper)

                e2 = {
                    "Alias/Nom": f"{f_name}_Whisper",
                    "Méthode": "Whisper Large",
                    "Modèle": "whisper-large",
                    "Durée": human_time(orig_sec),
                    "Temps": human_time(orig_sec),
                    "Coût": "?",
                    "Transcription": txt_whisper,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(e2)

            else:
                # Transcription simple
                chosen_model = "nova-2" if model_selection=="Nova 2" else "whisper-large"
                txt = try_transcription(temp_filename, keys, language_selection, chosen_model)
                st.subheader(f"[{model_selection}] {f_name}")
                st.text_area("Résultat", txt, height=150)
                copy_to_clipboard(txt)

                eX = {
                    "Alias/Nom": f"{f_name}_{chosen_model}",
                    "Méthode": model_selection,
                    "Modèle": chosen_model,
                    "Durée": human_time(orig_sec),
                    "Temps": human_time(orig_sec),
                    "Coût": "?",
                    "Transcription": txt,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(eX)

            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        display_history(st.session_state["history"])

        st.write("---")
        st.write("### Aperçu Audio")
        for en in st.session_state["history"]:
            st.audio(bytes.fromhex(en["Audio Binaire"]), format="audio/wav")


def main_wrapper():
    try:
        main()
    except Exception as e:
        st.error(f"Erreur : {e}")

if __name__=="__main__":
    main_wrapper()
