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

############################################
#  On place set_page_config COMME TOUT PREMIER APPEL
############################################
st.set_page_config(page_title="NBL Audio", layout="wide")

############################################
#          INITIALISATIONS ET UTILS        #
############################################

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
    """Récupère les clés API (NOVA...)."""
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def pick_key(keys):
    """Prend la clé courante, sans rotation multiple."""
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

def copy_to_clipboard(txt):
    script_html = f"""
    <script>
        function copyText(t) {{
            navigator.clipboard.writeText(t).then(function() {{
                alert('Transcription copiée !');
            }}, function(err) {{
                alert('Échec : ' + err);
            }});
        }}
    </script>
    <button onclick="copyText(`{txt}`)" style="margin:5px;">Copier</button>
    """
    st.components.v1.html(script_html)

def password_gate():
    st.title("Accès Protégé (Mot de Passe)")
    st.info("Veuillez saisir le code à 4 chiffres.")
    code_in = st.text_input("Code:", value="", max_chars=4, type="password", key="pwd_in")
    if st.button("Valider", key="pwd_valider"):
        if st.session_state.get("blocked", False):
            st.warning("Vous êtes déjà bloqué.")
            st.stop()
        if st.session_state.get("pwd_attempts", 0)>=4:
            st.error("Trop de tentatives. Session bloquée.")
            st.session_state["blocked"] = True
            st.stop()

        real_pwd = st.secrets.get("APP_PWD","1234")
        if code_in==real_pwd:
            st.session_state["authorized"] = True
            st.success("Accès autorisé.")
        else:
            st.session_state["pwd_attempts"] += 1
            attempts = st.session_state["pwd_attempts"]
            st.error(f"Mot de passe invalide. (Tentative {attempts}/4)")

############################################
#         APP PRINCIPALE  (APRES PWD)      #
############################################

def app_main():
    st.title("NBL Audio")  # Titre principal

    # Micro par défaut => index=1
    source_type = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"], index=1)
    audio_data_list = []
    file_names = []

    if source_type=="Fichier (Upload)":
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
        # Activation micro par défaut
        nb_mics = st.number_input("Nombre de micros", min_value=1, max_value=4, value=1)
        for i in range(nb_mics):
            mic_inp = st.audio_input(f"Micro {i+1}")
            if mic_inp:
                audio_data_list.append(mic_inp.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic_inp, format=mic_inp.type)

    st.write("---")
    st.subheader("Options de Transcription")
    # Explication Double Trans
    st.markdown("""
**Double Transcription** : 
- **Nova 2** (IA plus rapide),
- **Whisper Large** (IA plus précise ~99%).
    """)
    colA, colB = st.columns([1,1])
    with colA:
        model_main = st.selectbox("Modèle Principal", ["Nova 2","Whisper Large"])
    with colB:
        lang_main = "fr"
        if model_main=="Whisper Large":
            lang_main = st.selectbox("Langue Whisper", ["fr","en"])

    double_trans = st.checkbox("Activer Double Transcription (Nova 2 -> Whisper)")

    if st.button("Transcrire") and audio_data_list:
        st.info("Transcription en cours...")
        keys = get_dg_keys()
        if not keys:
            st.error("Aucune clé DeepGram disponible.")
            st.stop()

        for idx, raw_data in enumerate(audio_data_list):
            seg = AudioSegment.from_file(io.BytesIO(raw_data))
            dur_s = len(seg)/1000.0
            f_name = file_names[idx] if idx<len(file_names) else f"Fichier_{idx+1}"
            st.write(f"### Fichier {idx+1}: {f_name} / Durée={human_time(dur_s)}")

            # On crée un wav
            temp_wav = f"temp_input_{idx}.wav"
            seg.export(temp_wav, format="wav")

            if double_trans:
                # 1) Nova2
                start_nova = time.time()
                k_nova = pick_key(keys)
                txt_nova = nova_api.transcribe_audio(temp_wav, k_nova, "fr", "nova-2")
                end_nova = time.time()

                # 2) Whisper
                start_whisp = time.time()
                k_whisp = pick_key(keys)
                txt_whisp = nova_api.transcribe_audio(temp_wav, k_whisp, lang_main, "whisper-large")
                end_whisp = time.time()

                # Affichage côte à côte
                cL, cR = st.columns(2)
                with cL:
                    st.subheader(f"NOVA 2 - {f_name}")
                    st.text_area(f"Nova2_{f_name}", txt_nova, height=150, key=f"nova_{idx}")
                    copy_to_clipboard(txt_nova)
                    st.write(f"Temps: {end_nova - start_nova:.1f}s")
                with cR:
                    st.subheader(f"WHISPER LARGE - {f_name}")
                    st.text_area(f"Whisper_{f_name}", txt_whisp, height=150, key=f"whisp_{idx}")
                    copy_to_clipboard(txt_whisp)
                    st.write(f"Temps: {end_whisp - start_whisp:.1f}s")

                # Pas de tableau d'historique, mais on stocke si besoin
                eNova = {
                    "Alias/Nom": f"{f_name}_Nova2",
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": f"{human_time(dur_s)}",
                    "Temps": f"{(end_nova - start_nova):.1f}s",
                    "Coût": "?",
                    "Transcription": txt_nova,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                eWhisp = {
                    "Alias/Nom": f"{f_name}_Whisper",
                    "Méthode": "Whisper Large",
                    "Modèle": "whisper-large",
                    "Durée": f"{human_time(dur_s)}",
                    "Temps": f"{(end_whisp - start_whisp):.1f}s",
                    "Coût": "?",
                    "Transcription": txt_whisp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].extend([eNova,eWhisp])

            else:
                start_one = time.time()
                k_simp = pick_key(keys)
                chosen_model = "nova-2" if model_main=="Nova 2" else "whisper-large"
                chosen_lang = "fr" if chosen_model=="nova-2" else lang_main
                txt_simp = nova_api.transcribe_audio(temp_wav, k_simp, chosen_lang, chosen_model)
                end_one = time.time()

                # Affiche dans 1 col
                leftC, _ = st.columns([1,1])
                with leftC:
                    st.subheader(f"{model_main} - {f_name}")
                    st.text_area(f"Simple_{f_name}", txt_simp, height=150, key=f"area_{idx}_{model_main}")
                    copy_to_clipboard(txt_simp)
                    st.write(f"Temps: {end_one - start_one:.1f}s")

                eSimp = {
                    "Alias/Nom": f"{f_name}_{chosen_model}",
                    "Méthode": model_main,
                    "Modèle": chosen_model,
                    "Durée": f"{human_time(dur_s)}",
                    "Temps": f"{(end_one - start_one):.1f}s",
                    "Coût": "?",
                    "Transcription": txt_simp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(eSimp)

            if os.path.exists(temp_wav):
                os.remove(temp_wav)

def main():
    init_state()
    if st.session_state.get("blocked", False):
        st.error("Vous êtes bloqué pour cette session.")
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
