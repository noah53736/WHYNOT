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
import math

############################################
#         INITIALISATION ET UTILS          #
############################################

def init_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    # On ne tourne pas sur les clés: un index => si échec on passe à la suivante
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
    """ Récupère toutes les clés API depuis st.secrets. """
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def try_transcription(
    file_path: str,
    keys,
    language="fr",
    model="nova-2"
) -> str:
    """
    Tente de transcrire en utilisant la clé st.session_state["key_index"].
    Si échec => on passe à la suivante, etc.
    Retourne la transcription finale (string) ou "" en cas d'échec complet.
    """
    n = len(keys)
    attempt = 0
    idx = st.session_state["key_index"]
    while attempt < n:
        if idx >= n:
            idx = 0
        dg_key = keys[idx]
        idx += 1
        attempt += 1
        txt = nova_api.transcribe_audio(
            file_path,
            dg_key,
            language,
            model
        )
        if txt != "":
            st.session_state["key_index"] = idx-1
            return txt
    return ""

def segment_for_whisper(audio_seg, chunk_sec=20):
    """
    Segment audio en petits morceaux de `chunk_sec` secondes 
    pour éviter les limites de taille, 
    SANS couper au milieu d'une phrase (best effort).
    """
    # Approche simpliste : on coupe toutes les chunk_sec
    # (Améliorer si vous voulez éviter de couper les mots)
    segments = []
    length_ms = len(audio_seg)
    step = chunk_sec * 1000
    for start in range(0, length_ms, step):
        end = min(start+step, length_ms)
        segments.append(audio_seg[start:end])
    return segments

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
        st.error("Pas de clés API config.")
        st.stop()

    # 1) Entrée Audio
    st.subheader("Entrée Audio")
    audio_data = []
    file_names = []

    input_type = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"])
    if input_type=="Fichier (Upload)":
        upf = st.file_uploader(
            "Importer un ou plusieurs fichiers audio",
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
        nmics = st.number_input("Nombre de micros :", min_value=1, max_value=max_mics, value=1)
        for i in range(nmics):
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
            lang_sel = st.selectbox("Langue pour Whisper", ["fr","en"])
        else:
            lang_sel = "fr"

    double_trans = st.checkbox("Double Transcription (Nova2 + Whisper)")
    chunk_sec_input = st.slider("Segmentation (secondes)", 15,90,20)

    # 3) BOUTON
    if st.button("Transcrire") and audio_data:
        st.info("Début de la transcription...")
        for i, raw in enumerate(audio_data):
            file_name = file_names[i] if i<len(file_names) else f"Fichier_{i+1}"
            seg = AudioSegment.from_file(io.BytesIO(raw))
            orig_s = len(seg)/1000.0

            st.write(f"**Fichier {i+1} : {file_name}** / Durée={human_time(orig_s)}")

            # Condition segmentation si Whisper Large
            do_seg = False
            if model_selection=="Whisper Large" or double_trans:
                # On segmente si le fichier est grand
                if len(raw)>20*1024*1024:
                    do_seg = True

            # segment
            if do_seg:
                st.info("Segmentation en cours...")
                segs = segment_for_whisper(seg, chunk_sec=chunk_sec_input)
            else:
                segs = [seg]

            # Exporter en .wav
            chunk_files = []
            for s_idx, s_seg in enumerate(segs):
                tmp_out = f"temp_chunk_{i}_{s_idx}.wav"
                s_seg.export(tmp_out, format="wav")
                chunk_files.append(tmp_out)

            if double_trans:
                # On fait Nova2 puis Whisper
                # 1) On fait Nova2 en bloc => on concat
                t_nova_list = []
                total_nova_time_start = time.time()
                for cf in chunk_files:
                    txtn = try_transcription(cf, keys, "fr", "nova-2")
                    t_nova_list.append(txtn)
                    os.remove(cf)
                total_nova_time = time.time() - total_nova_time_start

                final_nova = " ".join(t_nova_list)
                # On affiche tout de suite
                leftC, rightC = st.columns([1,1])
                with leftC:
                    st.subheader(f"{file_name} - NOVA 2")
                    st.text_area("Transcription Nova 2", final_nova, height=150)
                    copy_to_clipboard(final_nova)

                # On ajoute à l'historique
                eNova = {
                    "Alias/Nom": f"{file_name}_NOVA2",
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": human_time(orig_s),
                    "Temps": f"{total_nova_time:.1f}s",
                    "Coût": "?",
                    "Transcription": final_nova,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw.hex()
                }
                st.session_state["history"].append(eNova)

                # 2) On fait WhisperLarge
                # On refait la segmentation .wav
                chunk_files2 = []
                for s_idx, s_seg2 in enumerate(segs):
                    tmp_out2 = f"temp_chunk2_{i}_{s_idx}.wav"
                    s_seg2.export(tmp_out2, format="wav")
                    chunk_files2.append(tmp_out2)

                t_whisper_list = []
                total_whisper_time_start = time.time()
                for cf2 in chunk_files2:
                    txtw = try_transcription(cf2, keys, lang_sel, "whisper-large")
                    t_whisper_list.append(txtw)
                    os.remove(cf2)
                total_whisper_time = time.time() - total_whisper_time_start

                final_whisper = " ".join(t_whisper_list)
                with rightC:
                    st.subheader(f"{file_name} - Whisper Large")
                    st.text_area("Transcription Whisper", final_whisper, height=150)
                    copy_to_clipboard(final_whisper)

                eWhisp = {
                    "Alias/Nom": f"{file_name}_WHISPER",
                    "Méthode": "Whisper Large",
                    "Modèle": "whisper-large",
                    "Durée": human_time(orig_s),
                    "Temps": f"{total_whisper_time:.1f}s",
                    "Coût": "?",
                    "Transcription": final_whisper,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw.hex()
                }
                st.session_state["history"].append(eWhisp)

            else:
                # Simple transcription
                chosen_model = "nova-2" if model_selection=="Nova 2" else "whisper-large"
                model_tag = model_selection
                # On transcrit tout
                t_list = []
                total_time_start = time.time()
                for cf in chunk_files:
                    txt_simp = try_transcription(cf, keys, lang_sel if chosen_model=="whisper-large" else "fr", chosen_model)
                    t_list.append(txt_simp)
                    os.remove(cf)
                total_time = time.time() - total_time_start
                final_txt = " ".join(t_list)

                leftC, _ = st.columns([1,1])
                with leftC:
                    st.subheader(f"{file_name} - {model_selection}")
                    st.text_area("Transcription", final_txt, height=150)
                    copy_to_clipboard(final_txt)

                eSimp = {
                    "Alias/Nom": f"{file_name}_{chosen_model}",
                    "Méthode": model_tag,
                    "Modèle": chosen_model,
                    "Durée": human_time(orig_s),
                    "Temps": f"{total_time:.1f}s",
                    "Coût": "?",
                    "Transcription": final_txt,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw.hex()
                }
                st.session_state["history"].append(eSimp)

        # Fin
        display_history(st.session_state["history"])
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
