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
#    INITIALISATIONS    #
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
    """Récupère les clés API DeepGram depuis st.secrets (NOVA...)."""
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def pick_key(keys):
    """
    On ne fait pas de rotation multiple.
    On prend la clé courante, si on dépasse, on revient au 1er index.
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
#     MOT DE PASSE      #
#########################
def password_gate():
    st.title("Accès Protégé (Mot de Passe)")
    st.info("Veuillez saisir le code à 4 chiffres.")
    code_in = st.text_input("Code:", value="", max_chars=4, type="password", key="pwd_in")
    if st.button("Valider", key="pwd_valider"):
        if st.session_state.get("blocked", False):
            st.warning("Vous êtes bloqué.")
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

#########################
#   SEGMENTATION 25MB   #
#########################
def chunk_if_needed(raw_data: bytes, max_size=25*1024*1024):
    """
    Découpe en blocs de ~25MB si trop volumineux pour Whisper Large.
    Pas de prise en compte de la phrase.
    """
    if len(raw_data)<=max_size:
        return [AudioSegment.from_file(io.BytesIO(raw_data))]
    st.info("Fichier volumineux => segmentation 25MB (Whisper Large).")
    seg_full = AudioSegment.from_file(io.BytesIO(raw_data))
    segments = []
    length_ms = len(seg_full)
    # proportionnel
    step_ms = int(length_ms*(max_size/len(raw_data)))
    start=0
    while start<length_ms:
        end = min(start+step_ms, length_ms)
        segments.append(seg_full[start:end])
        start=end
    return segments

#########################
#  LOGIQUE DE L'APP     #
#########################
def app_main():
    st.title("NBL Audio")

    st.subheader("Source Audio (Micro par défaut)")
    # Micro par défaut => index=1
    source = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"], index=1)
    audio_data_list = []
    file_names = []

    if source=="Fichier (Upload)":
        upf = st.file_uploader(
            "Fichiers audio",
            type=["mp3","wav","m4a","ogg","webm"],
            accept_multiple_files=True
        )
        if upf:
            for f in upf:
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
**Double Transcription** (par défaut) => 2 IA successives :
- **Nova 2** (rapide)
- **Whisper Large** (précis, mais plus lent)
    """)
    double_trans = st.checkbox("Double Transcription (Nova2 -> Whisper)", value=True)

    colA,colB = st.columns([1,1])
    with colA:
        model_main = st.selectbox("Modèle Unique (si double non coché)", ["Nova 2","Whisper Large"])
    with colB:
        lang_main = "fr"
        if model_main=="Whisper Large":
            lang_main = st.selectbox("Langue Whisper", ["fr","en"])

    if st.button("Transcrire") and audio_data_list:
        st.info("Transcription en cours...")
        keys = get_dg_keys()
        if not keys:
            st.error("Pas de clé DeepGram disponible.")
            st.stop()

        for idx, raw_data in enumerate(audio_data_list):
            seg = AudioSegment.from_file(io.BytesIO(raw_data))
            dur_s = len(seg)/1000.0
            f_name = file_names[idx] if idx<len(file_names) else f"Fichier_{idx+1}"

            st.write(f"### Fichier {idx+1} : {f_name} (Durée: {human_time(dur_s)})")

            # On décide si on segmente seulement si (Whisper) ET trop volumineux
            do_seg = False
            if (double_trans or model_main=="Whisper Large") and len(raw_data)>25*1024*1024:
                do_seg = True

            if do_seg:
                segs = chunk_if_needed(raw_data, max_size=25*1024*1024)
            else:
                segs = [seg]

            chunk_files = []
            for s_i, s_seg in enumerate(segs):
                tmp_wav = f"temp_segment_{idx}_{s_i}.wav"
                s_seg.export(tmp_wav, format="wav")
                chunk_files.append(tmp_wav)

            if double_trans:
                # 1) Nova 2 => plus rapide
                st.write(f"**{f_name} => Nova 2** (plus rapide)")
                start_nova = time.time()
                txt_nova_list = []
                for cf in chunk_files:
                    k_nova = pick_key(keys)
                    txt_n = nova_api.transcribe_audio(cf, k_nova, "fr", "nova-2")
                    txt_nova_list.append(txt_n)
                end_nova = time.time()
                final_nova = " ".join(txt_nova_list)

                # On affiche direct
                st.success("Nova 2 terminé, en attente de Whisper Large...")

                # 2) Whisper Large => plus lent
                st.write(f"**{f_name} => Whisper Large** (plus précis ~99%)")
                start_whisp = time.time()
                txt_whisp_list = []
                for cf2 in chunk_files:
                    k_whisp = pick_key(keys)
                    txt_w = nova_api.transcribe_audio(cf2, k_whisp, lang_main, "whisper-large")
                    txt_whisp_list.append(txt_w)
                end_whisp = time.time()
                final_whisp = " ".join(txt_whisp_list)

                # Affichage
                cL, cR = st.columns(2)
                with cL:
                    st.subheader(f"NOVA 2 - {f_name}")
                    st.text_area(
                        f"Nova2_{f_name}", 
                        final_nova,
                        key=f"nova2_{idx}",
                        height=150
                    )
                    copy_to_clipboard(final_nova)
                    st.write(f"Temps Nova 2: {end_nova - start_nova:.1f}s")

                with cR:
                    st.subheader(f"WHISPER LARGE - {f_name}")
                    st.text_area(
                        f"Whisper_{f_name}",
                        final_whisp,
                        key=f"whisper_{idx}",
                        height=150
                    )
                    copy_to_clipboard(final_whisp)
                    st.write(f"Temps Whisper: {end_whisp - start_whisp:.1f}s")

                # On stocke
                eNova = {
                    "Alias/Nom": f"{f_name}_Nova2",
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": f"{human_time(dur_s)}",
                    "Temps": f"{(end_nova - start_nova):.1f}s",
                    "Coût": "?",
                    "Transcription": final_nova,
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
                    "Transcription": final_whisp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].extend([eNova,eWhisp])

            else:
                # Simple
                st.write(f"**{f_name} => Mode Simple**")
                start_simp = time.time()
                final_list = []
                chosen_model = "nova-2" if model_main=="Nova 2" else "whisper-large"
                chosen_lang = "fr" if chosen_model=="nova-2" else lang_main
                for cf3 in chunk_files:
                    k_simp = pick_key(keys)
                    tx_s = nova_api.transcribe_audio(cf3, k_simp, chosen_lang, chosen_model)
                    final_list.append(tx_s)
                end_simp = time.time()
                final_simp = " ".join(final_list)

                leftC, _ = st.columns([1,1])
                with leftC:
                    st.subheader(f"{model_main} - {f_name}")
                    st.text_area(
                        f"Simple_{f_name}",
                        final_simp,
                        key=f"simple_{idx}",
                        height=150
                    )
                    copy_to_clipboard(final_simp)
                    st.write(f"Temps: {end_simp - start_simp:.1f}s")

                eSimp = {
                    "Alias/Nom": f"{f_name}_{chosen_model}",
                    "Méthode": model_main,
                    "Modèle": chosen_model,
                    "Durée": f"{human_time(dur_s)}",
                    "Temps": f"{(end_simp - start_simp):.1f}s",
                    "Coût": "?",
                    "Transcription": final_simp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(eSimp)

            for cf4 in chunk_files:
                if os.path.exists(cf4):
                    os.remove(cf4)

def main():
    init_state()
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
