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
    # Historique "whisper-only"
    if "whisper_history" not in st.session_state:
        st.session_state["whisper_history"] = []
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
    On prend la clé courante, 
    si on dépasse => retour index 0.
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

###############################
#  AFFICHAGE HISTO WHISPER    #
###############################
def display_whisper_history():
    st.sidebar.write("---")
    st.sidebar.header("Historique Whisper Only")
    if st.session_state["whisper_history"]:
        table = []
        for wh in st.session_state["whisper_history"]:
            table.append({
                "Nom": wh["Alias/Nom"],
                "Durée": wh["Durée"],
                "Temps": wh["Temps"],
                "Transcrit?": (wh["Transcription"][:30] + "...") if wh["Transcription"] else ""
            })
        st.sidebar.table(table[::-1])
        if st.sidebar.button("Vider Historique Whisper", key="clear_whisp"):
            st.session_state["whisper_history"].clear()
            st.experimental_rerun()
    else:
        st.sidebar.info("Aucun historique Whisper.")

#########################
#     APP PRINCIPALE    #
#########################
def app_main():
    display_whisper_history()
    st.title("NBL Audio")

    # Micro par défaut => index=1
    src_type = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"], index=1)
    audio_data_list = []
    file_names = []

    if src_type=="Fichier (Upload)":
        upfs = st.file_uploader(
            "Importer vos fichiers audio",
            type=["mp3","wav","m4a","ogg","webm"],
            accept_multiple_files=True
        )
        if upfs:
            for f in upfs:
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
2. Whisper Large (plus lent, ~99% précision)
""")
    double_trans = st.checkbox("Double Transcription (Nova2 -> Whisper)", value=True)

    colA,colB = st.columns([1,1])
    with colA:
        model_main = st.selectbox("Modèle (unique si pas double)", ["Nova 2","Whisper Large"])
    with colB:
        lang_main = "fr"
        if model_main=="Whisper Large":
            lang_main = st.selectbox("Langue (Whisper):", ["fr","en"])

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

            st.write(f"### Fichier {idx+1} : {f_name} (Durée: {human_time(dur_s)})")

            temp_wav = f"temp_input_{idx}.wav"
            seg.export(temp_wav, format="wav")

            if double_trans:
                # 1) Nova 2
                st.write(f"**{f_name} => Nova 2** (rapide)")
                start_nova = time.time()
                key_nova = pick_key(keys)
                txt_nova = nova_api.transcribe_audio(temp_wav, key_nova, "fr", "nova-2")
                end_nova = time.time()

                # Affiche direct le Nova
                st.success(f"Nova 2 terminé pour {f_name}, en attente de WhisperLarge...")
                leftC, rightC = st.columns(2)
                with leftC:
                    st.subheader(f"NOVA 2 - {f_name}")
                    st.text_area(
                        f"Nova_{f_name}",
                        txt_nova,
                        key=f"nova_{idx}",
                        height=120
                    )
                    copy_to_clipboard(txt_nova)
                    st.write(f"Temps Nova : {end_nova - start_nova:.1f}s")

                # 2) Whisper Large
                st.info(f"Début Whisper Large pour {f_name}...")
                start_whisp = time.time()
                key_whisp = pick_key(keys)
                txt_whisp = nova_api.transcribe_audio(temp_wav, key_whisp, lang_main, "whisper-large")
                end_whisp = time.time()

                with rightC:
                    st.subheader(f"WHISPER LARGE - {f_name}")
                    st.text_area(
                        f"Whisp_{f_name}",
                        txt_whisp,
                        key=f"whisp_{idx}",
                        height=120
                    )
                    copy_to_clipboard(txt_whisp)
                    st.write(f"Temps Whisper : {end_whisp - start_whisp:.1f}s")

                # On stocke l'historique whisper
                eWhisp = {
                    "Alias/Nom": f"{f_name}_Whisper",
                    "Durée": f"{human_time(dur_s)}",
                    "Temps": f"{(end_whisp - start_whisp):.1f}s",
                    "Transcription": txt_whisp
                }
                st.session_state["whisper_history"].append(eWhisp)

            else:
                # Simple
                start_simp = time.time()
                key_simp = pick_key(keys)
                chosen_model = "nova-2" if model_main=="Nova 2" else "whisper-large"
                chosen_lang = "fr" if chosen_model=="nova-2" else lang_main
                txt_simp = nova_api.transcribe_audio(temp_wav, key_simp, chosen_lang, chosen_model)
                end_simp = time.time()

                cA, _ = st.columns([1,1])
                with cA:
                    st.subheader(f"{model_main} - {f_name}")
                    st.text_area(
                        f"Result_{f_name}",
                        txt_simp,
                        key=f"simple_{idx}",
                        height=120
                    )
                    copy_to_clipboard(txt_simp)
                    st.write(f"Temps: {end_simp - start_simp:.1f}s")

                # Si c'est Whisper Large, on stocke
                if chosen_model=="whisper-large":
                    eW = {
                        "Alias/Nom": f"{f_name}_Whisper",
                        "Durée": f"{human_time(dur_s)}",
                        "Temps": f"{(end_simp - start_simp):.1f}s",
                        "Transcription": txt_simp
                    }
                    st.session_state["whisper_history"].append(eW)

            if os.path.exists(temp_wav):
                os.remove(temp_wav)

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
