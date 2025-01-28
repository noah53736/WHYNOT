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

def init_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
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

def get_api_keys():
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def try_transcription(file_path, keys, language="fr", model="nova-2"):
    idx = st.session_state["key_index"]
    n = len(keys)
    attempt = 0
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

def chunk_if_needed(data: bytes, max_size=25*1024*1024):
    if len(data) <= max_size:
        return [AudioSegment.from_file(io.BytesIO(data))]
    st.info("Fichier volumineux, segmentation 25MB.")
    big_seg = AudioSegment.from_file(io.BytesIO(data))
    out = []
    start = 0
    step_ms = int(len(big_seg)*(max_size/len(data)))
    while start < len(big_seg):
        end = min(start+step_ms, len(big_seg))
        out.append(big_seg[start:end])
        start = end
    return out

def main():
    st.set_page_config(page_title="NBL Audio", layout="wide")
    st.title("NBL Audio")

    init_state()
    history = st.session_state["history"]
    keys = get_api_keys()
    if not keys:
        st.error("Pas de clés API configurées.")
        st.stop()

    st.subheader("Entrée Audio")
    audio_data = []
    file_names = []
    source_type = st.radio("Source", ["Fichier (Upload)", "Micro (Enregistrement)"])
    if source_type=="Fichier (Upload)":
        upfs = st.file_uploader("Fichiers audio", type=["mp3","wav","m4a","ogg","webm"], accept_multiple_files=True)
        if upfs:
            for f in upfs:
                audio_data.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        nb_mics = st.number_input("Nb micros", min_value=1, max_value=4, value=1)
        for i in range(nb_mics):
            mic = st.audio_input(f"Micro {i+1}")
            if mic:
                audio_data.append(mic.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic, format=mic.type)

    st.write("---")
    st.subheader("Options")
    col1, col2 = st.columns([1,1])
    with col1:
        model_main = st.selectbox("Modèle Principal", ["Nova 2","Whisper Large"])
    with col2:
        lang_main = "fr"
        if model_main=="Whisper Large":
            lang_main = st.selectbox("Langue", ["fr","en"])

    double_trans = st.checkbox("Double Transcription (Nova puis Whisper)")

    if st.button("Transcrire") and audio_data:
        for idx, raw in enumerate(audio_data):
            st.write(f"### Fichier {idx+1}")
            segs = chunk_if_needed(raw, 25*1024*1024)
            file_name = file_names[idx] if idx<len(file_names) else f"F_{idx+1}"
            original_seg = AudioSegment.from_file(io.BytesIO(raw))
            original_sec = len(original_seg)/1000.0

            if double_trans:
                # 1) Nova
                st.write(f"**{file_name}: Transcription Nova 2**")
                start_nova = time.time()
                partial_nova = []
                for s_i, subseg in enumerate(segs):
                    tmp_wav = f"temp_nova_{idx}_{s_i}.wav"
                    subseg.export(tmp_wav, format="wav")
                    txt = try_transcription(tmp_wav, keys, "fr", "nova-2")
                    partial_nova.append(txt)
                    os.remove(tmp_wav)
                final_nova = " ".join(partial_nova)
                leftC, rightC = st.columns([1,1])
                with leftC:
                    st.subheader("Nova 2")
                    st.text_area(f"Résultat - {file_name}", final_nova, height=120)
                    copy_to_clipboard(final_nova)
                time_nova = time.time()-start_nova

                eNova = {
                    "Alias/Nom": f"{file_name}_NOVA2",
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": human_time(original_sec),
                    "Temps": f"{time_nova:.1f}s",
                    "Coût": "?",
                    "Transcription": final_nova,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw.hex()
                }
                st.session_state["history"].append(eNova)

                # 2) Whisper
                st.write(f"**{file_name}: Transcription Whisper Large**")
                start_whisp = time.time()
                partial_whisp = []
                for s_j, subsegW in enumerate(segs):
                    tmp_wav2 = f"temp_whisp_{idx}_{s_j}.wav"
                    subsegW.export(tmp_wav2, format="wav")
                    txtW = try_transcription(tmp_wav2, keys, lang_main, "whisper-large")
                    partial_whisp.append(txtW)
                    os.remove(tmp_wav2)
                final_whisp = " ".join(partial_whisp)
                with rightC:
                    st.subheader("Whisper Large")
                    st.text_area(f"Résultat - {file_name}", final_whisp, height=120)
                    copy_to_clipboard(final_whisp)
                time_whisp = time.time()-start_whisp

                eWhisp = {
                    "Alias/Nom": f"{file_name}_WHISPER",
                    "Méthode": "Whisper Large",
                    "Modèle": "whisper-large",
                    "Durée": human_time(original_sec),
                    "Temps": f"{time_whisp:.1f}s",
                    "Coût": "?",
                    "Transcription": final_whisp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw.hex()
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
                start_simp = time.time()
                partial_simp = []
                for s_k, subsegS in enumerate(segs):
                    tmp_simp = f"temp_simp_{idx}_{s_k}.wav"
                    subsegS.export(tmp_simp, format="wav")
                    tS = try_transcription(tmp_simp, keys, chosen_lang, chosen_model)
                    partial_simp.append(tS)
                    os.remove(tmp_simp)
                final_simp = " ".join(partial_simp)
                leftC, _ = st.columns([1,1])
                with leftC:
                    st.subheader(f"{model_main}")
                    st.text_area(f"Résultat - {file_name}", final_simp, height=120)
                    copy_to_clipboard(final_simp)
                time_simp = time.time()-start_simp
                eSimp = {
                    "Alias/Nom": f"{file_name}_{chosen_model}",
                    "Méthode": model_main,
                    "Modèle": chosen_model,
                    "Durée": human_time(original_sec),
                    "Temps": f"{time_simp:.1f}s",
                    "Coût": "?",
                    "Transcription": final_simp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw.hex()
                }
                st.session_state["history"].append(eSimp)

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
