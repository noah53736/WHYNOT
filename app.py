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

#########################
#   CONFIG PAGE UNIQUE  #
#########################
st.set_page_config(page_title="NBL Audio", layout="wide")

#########################
#   INITIALISATIONS     #
#########################
def init_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "dg_key_index" not in st.session_state:
        st.session_state["dg_key_index"] = 0
    if "pwd_attempts" not in st.session_state:
        st.session_state["pwd_attempts"] = 0
    if "authorized" not in st.session_state:
        st.session_state["authorized"] = False
    if "blocked" not in st.session_state:
        st.session_state["blocked"] = False

def get_dg_keys():
    """Récupère les clés API DeepGram (NOVA...)."""
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def pick_key(keys):
    """
    On ne fait pas de rotation multiple.
    On prend la clé courante, si on dépasse => retour index 0.
    """
    if not keys:
        return None
    idx = st.session_state["dg_key_index"]
    if idx >= len(keys):
        st.session_state["dg_key_index"] = 0
        idx = 0
    return keys[idx]

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
                alert('Transcription copiée !');
            }}, function(err) {{
                alert('Échec : ' + err);
            }});
        }}
    </script>
    <button onclick="copyText(`{text}`)" style="margin:5px;">Copier</button>
    """
    st.components.v1.html(script_html)

#########################
#      MOT DE PASSE     #
#########################
def password_gate():
    st.title("Accès Protégé (Mot de Passe)")
    st.info("Veuillez saisir le code à 4 chiffres.")
    code_in = st.text_input("Code:", value="", max_chars=4, type="password", key="pwd_in")
    if st.button("Valider", key="pwd_valider"):
        if st.session_state.get("blocked", False):
            st.warning("Vous êtes bloqué.")
            st.stop()
        if st.session_state.get("pwd_attempts", 0) >= 4:
            st.error("Trop de tentatives. Session bloquée.")
            st.session_state["blocked"] = True
            st.stop()

        real_pwd = st.secrets.get("APP_PWD","1234")
        if code_in == real_pwd:
            st.session_state["authorized"] = True
            st.success("Accès autorisé.")
        else:
            st.session_state["pwd_attempts"] += 1
            attempts = st.session_state["pwd_attempts"]
            st.error(f"Mot de passe invalide. (Tentative {attempts}/4)")

#########################
# SEGMENTATION (22 MB)  #
#########################
def segment_if_needed(raw_data: bytes, threshold=22*1024*1024):
    """
    Segmente le fichier en ~22MB si c'est plus grand,
    UNIQUEMENT pour Whisper Large usage.
    """
    if len(raw_data) <= threshold:
        return [AudioSegment.from_file(io.BytesIO(raw_data))]
    st.info("Fichier volumineux => segmentation par blocs de 22MB (Whisper Large).")
    full_seg = AudioSegment.from_file(io.BytesIO(raw_data))
    segments = []
    length_ms = len(full_seg)
    # proportionnel
    step_ms = int(length_ms * (threshold/len(raw_data)))
    start = 0
    while start < length_ms:
        end = min(start+step_ms, length_ms)
        segments.append(full_seg[start:end])
        start = end
    return segments

#########################
#   FONCTION TRANSCRIBE #
#########################
def transcribe_chunklist(chunk_files, model, lang, keys):
    """
    Transcrit successivement chaque chunk 'chunk_files' (WAV)
    et assemble le résultat.
    """
    texts = []
    for cf in chunk_files:
        k = pick_key(keys)
        txt = nova_api.transcribe_audio(cf, k, lang, model)
        texts.append(txt)
    return " ".join(texts)

#########################
#     APP PRINCIPALE    #
#########################
def app_main():
    st.title("NBL Audio")

    # Micro par défaut => index=1
    src_type = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"], index=1)
    audio_data_list = []
    file_names = []

    if src_type=="Fichier (Upload)":
        uploaded_files = st.file_uploader(
            "Importer vos fichiers audio",
            type=["mp3","wav","m4a","ogg","webm"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for f in uploaded_files:
                audio_data_list.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        nmics = st.number_input("Nombre de micros", min_value=1, max_value=4, value=1)
        for i in range(nmics):
            mic_inp = st.audio_input(f"Micro {i+1}")
            if mic_inp:
                audio_data_list.append(mic_inp.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic_inp, format=mic_inp.type)

    st.write("---")
    st.subheader("Options de Transcription")
    st.markdown("""
**Double Transcription** (par défaut): 
1. Nova 2 (rapide)
2. Whisper Large (précise ~99%)
""")
    double_trans = st.checkbox("Double Transcription (Nova2 -> Whisper)", value=True)

    colA, colB = st.columns([1,1])
    with colA:
        model_single = st.selectbox("Modèle Unique", ["Nova 2","Whisper Large"])
    with colB:
        lang_single = "fr"
        if model_single=="Whisper Large":
            lang_single = st.selectbox("Langue (Whisper)", ["fr","en"])

    if st.button("Transcrire") and audio_data_list:
        st.info("Transcription en cours...")
        keys = get_dg_keys()
        if not keys:
            st.error("Pas de clé API disponible.")
            return

        for idx, raw_data in enumerate(audio_data_list):
            seg = AudioSegment.from_file(io.BytesIO(raw_data))
            dur_s = len(seg)/1000.0
            f_name = file_names[idx] if idx<len(file_names) else f"Fichier_{idx+1}"
            st.write(f"### Fichier {idx+1}: {f_name} (Durée {human_time(dur_s)})")

            # Détection si besoin segmentation
            # => SEULEMENT si Whisper est utilisé (double trans => whisper ou single => whisper)
            # => ET plus de 22 MB
            need_seg = False
            if double_trans:
                # Dans double trans => whisper always used
                if len(raw_data) > (22*1024*1024):
                    need_seg = True
            else:
                if model_single=="Whisper Large" and len(raw_data)>(22*1024*1024):
                    need_seg = True

            # On segmente si besoin
            if need_seg:
                segments = segment_if_needed(raw_data, threshold=22*1024*1024)
            else:
                segments = [seg]

            # Crée des fichiers .wav => on envoie
            chunk_files = []
            for s_i, s_seg in enumerate(segments):
                tmpf = f"tmp_{idx}_{s_i}.wav"
                s_seg.export(tmpf, format="wav")
                chunk_files.append(tmpf)

            if double_trans:
                # 1) Nova 2
                st.write(f"**{f_name} => Nova 2 (rapide)**")
                st.info("Démarrage Nova 2...")
                start_nova = time.time()
                final_nova = transcribe_chunklist(chunk_files, "nova-2", "fr", keys)
                end_nova = time.time()

                st.success("Nova 2 fini => en attente de Whisper Large...")

                # 2) Whisper Large
                st.write(f"**{f_name} => Whisper Large (plus précise)**")
                st.info("Démarrage Whisper Large...")
                start_whisp = time.time()
                final_whisp = transcribe_chunklist(chunk_files, "whisper-large", lang_single, keys)
                end_whisp = time.time()

                # AFFICHAGE
                cL, cR = st.columns(2)
                with cL:
                    st.subheader(f"{f_name} - Nova 2")
                    st.text_area(
                        f"Nova2_{f_name}",
                        final_nova,
                        key=f"nova2_{idx}",
                        height=150
                    )
                    copy_to_clipboard(final_nova)
                    st.write(f"Temps Nova2: {end_nova - start_nova:.1f}s")

                with cR:
                    st.subheader(f"{f_name} - Whisper Large")
                    st.text_area(
                        f"Whisp_{f_name}",
                        final_whisp,
                        key=f"whisp_{idx}",
                        height=150
                    )
                    copy_to_clipboard(final_whisp)
                    st.write(f"Temps Whisper: {end_whisp - start_whisp:.1f}s")

            else:
                # Mode unique => soit Nova2 soit Whisper
                chosen_model = "nova-2" if model_single=="Nova 2" else "whisper-large"
                chosen_lang = "fr" if chosen_model=="nova-2" else lang_single
                st.write(f"**{f_name} => {model_single}**")
                start_simp = time.time()
                final_simp = transcribe_chunklist(chunk_files, chosen_model, chosen_lang, keys)
                end_simp = time.time()

                leftC, _ = st.columns([1,1])
                with leftC:
                    st.subheader(f"{f_name} - {model_single}")
                    st.text_area(
                        f"simple_{f_name}",
                        final_simp,
                        key=f"textarea_{idx}_{model_single}",
                        height=150
                    )
                    copy_to_clipboard(final_simp)
                    st.write(f"Temps: {end_simp - start_simp:.1f}s")

            # Nettoyage
            for c_ in chunk_files:
                if os.path.exists(c_):
                    os.remove(c_)

def main():
    if "init" not in st.session_state:
        init_state()
        st.session_state["init"] = True

    if st.session_state.get("blocked",False):
        st.error("Vous êtes bloqué (trop de tentatives).")
        st.stop()
    if not st.session_state.get("authorized",False):
        password_gate()
        if st.session_state.get("authorized",False):
            app_main()
    else:
        app_main()

def main_wrapper():
    try:
        main()
    except Exception as e:
        st.error(f"Erreur Principale : {e}")

if __name__=="__main__":
    main_wrapper()
