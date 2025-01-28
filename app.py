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

#######################
#   INITIALISATIONS   #
#######################
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
    """Récupère les clés API depuis st.secrets (NOVA...)."""
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def pick_key(keys):
    """On prend la clé courante, index dg_key_index, sans rotation multiple."""
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

########################
#   MOT DE PASSE PAGE  #
########################
def password_protect():
    st.title("Accès Protégé")
    st.info("Veuillez entrer le code à 4 chiffres.")
    code_input = st.text_input("Code:", value="", max_chars=4, type="password")
    if st.button("Valider"):
        if st.session_state["pwd_attempts"]>=4:
            st.warning("Vous avez atteint le nombre maximum de tentatives.")
            st.session_state["blocked"] = True
            return
        app_pwd = st.secrets.get("APP_PWD", "1234")  # fallback
        if code_input == app_pwd:
            st.session_state["authorized"] = True
            st.success("Mot de passe correct, accès autorisé.")
        else:
            st.session_state["pwd_attempts"] += 1
            att = st.session_state["pwd_attempts"]
            st.error(f"Mot de passe incorrect (Tentative {att}/4).")

############################
#   SEGMENTATION SI LARGE  #
############################
def chunk_if_needed(raw_data: bytes, max_size=25*1024*1024):
    """
    Si le fichier dépasse 'max_size', on segmente en 25MB approx.
    SANS manip avancée, on coupe en blocs uniformes. 
    Pas de prise en compte des phrases.
    """
    if len(raw_data)<=max_size:
        return [AudioSegment.from_file(io.BytesIO(raw_data))]
    st.info("Segmentation du fichier car trop volumineux pour Whisper Large.")
    full_seg = AudioSegment.from_file(io.BytesIO(raw_data))
    segments = []
    step_ms = int(len(full_seg)*(max_size/len(raw_data)))
    start = 0
    while start<len(full_seg):
        end = min(start+step_ms, len(full_seg))
        segments.append(full_seg[start:end])
        start = end
    return segments

#######################
#  LOGIQUE PRINCIPALE #
#######################
def app_main():
    st.set_page_config(page_title="NBL Audio", layout="wide")
    st.title("NBL Audio")

    # Micro par défaut => radio index=1
    source_type = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"], index=1)
    audio_data_list = []
    file_names = []

    if source_type=="Fichier (Upload)":
        upfs = st.file_uploader("Importer des fichiers audio", 
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
            mic = st.audio_input(f"Micro {i+1}")
            if mic:
                audio_data_list.append(mic.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic, format=mic.type)

    st.write("---")
    st.subheader("Options de Transcription")
    # Double Trans plus didactique
    st.markdown("""**Double Transcription** : Utilise 2 IA successives :
- Nova 2 : plus rapide, 
- Whisper Large : plus précise (jusqu'à 99%).""")

    col1, col2 = st.columns([1,1])
    with col1:
        model_choice = st.selectbox("Modèle Principal :", ["Nova 2", "Whisper Large"])
    with col2:
        lang_main = "fr"
        if model_choice=="Whisper Large":
            lang_main = st.selectbox("Langue pour Whisper", ["fr","en"])

    double_trans = st.checkbox("Activer Double Transcription (Nova puis Whisper)")

    if st.button("Transcrire") and audio_data_list:
        st.info("Démarrage de la transcription...")
        dg_keys = get_dg_keys()
        if not dg_keys:
            st.error("Pas de clé API configurée.")
            st.stop()

        for idx, raw_data in enumerate(audio_data_list):
            f_name = file_names[idx] if idx<len(file_names) else f"Fichier_{idx+1}"
            audio_seg = AudioSegment.from_file(io.BytesIO(raw_data))
            dur_s = len(audio_seg)/1000.0
            st.write(f"### Fichier {idx+1}: {f_name}  (Durée: {human_time(dur_s)})")

            # Si Whisper Large ou double => on segmente si trop grand
            do_segment = False
            if (model_choice=="Whisper Large" or double_trans) and len(raw_data)>25*1024*1024:
                do_segment = True

            if do_segment:
                # On segmente
                segs = chunk_if_needed(raw_data, max_size=25*1024*1024)
            else:
                # Unique
                segs = [audio_seg]

            # Exporter en .wav => 1 ou plusieurs
            chunk_files = []
            for s_i, s_seg in enumerate(segs):
                tmpf = f"temp_{idx}_{s_i}.wav"
                s_seg.export(tmpf, format="wav")
                chunk_files.append(tmpf)

            if double_trans:
                # 1) Nova
                st.write(f"**{f_name} => Nova 2**")
                start_nova = time.time()
                txt_nova_list = []
                for cf in chunk_files:
                    key_nova = pick_key(dg_keys)
                    txt_nova_piece = nova_api.transcribe_audio(cf, key_nova, "fr", "nova-2")
                    txt_nova_list.append(txt_nova_piece)
                end_nova = time.time()
                final_nova = " ".join(txt_nova_list)

                # 2) Whisper
                st.write(f"**{f_name} => Whisper Large**")
                start_whisp = time.time()
                txt_whisp_list = []
                for cf2 in chunk_files:
                    key_whisp = pick_key(dg_keys)
                    txt_whisp_piece = nova_api.transcribe_audio(cf2, key_whisp, lang_main, "whisper-large")
                    txt_whisp_list.append(txt_whisp_piece)
                end_whisp = time.time()
                final_whisp = " ".join(txt_whisp_list)

                # Affichage côte à côte
                colL, colR = st.columns(2)
                with colL:
                    st.subheader(f"NOVA 2 - {f_name}")
                    st.text_area(f"Résultat Nova2 - {f_name}", final_nova, height=150, key=f"nova2_{idx}")
                    copy_to_clipboard(final_nova)
                    st.write(f"Temps: {end_nova - start_nova:.1f}s")

                with colR:
                    st.subheader(f"WHISPER LARGE - {f_name}")
                    st.text_area(f"Résultat Whisper - {f_name}", final_whisp, height=150, key=f"whisper_{idx}")
                    copy_to_clipboard(final_whisp)
                    st.write(f"Temps: {end_whisp - start_whisp:.1f}s")

                # Historique interne (pas affiché en tableau)
                eNova = {
                    "Alias/Nom": f"{f_name}_Nova2",
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": human_time(dur_s),
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
                    "Durée": human_time(dur_s),
                    "Temps": f"{(end_whisp - start_whisp):.1f}s",
                    "Coût": "?",
                    "Transcription": final_whisp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].extend([eNova,eWhisp])

            else:
                # Simple
                chosen_model = "nova-2" if model_choice=="Nova 2" else "whisper-large"
                chosen_lang = "fr" if chosen_model=="nova-2" else lang_main
                st.write(f"**{f_name} => {model_choice}**")
                start_simp = time.time()
                list_simp = []
                for cf3 in chunk_files:
                    key_simp = pick_key(dg_keys)
                    t_simp_piece = nova_api.transcribe_audio(cf3, key_simp, chosen_lang, chosen_model)
                    list_simp.append(t_simp_piece)
                end_simp = time.time()
                final_simp = " ".join(list_simp)

                leftC, _ = st.columns([1,1])
                with leftC:
                    st.subheader(f"{model_choice} - {f_name}")
                    st.text_area(
                        "Résultat",
                        final_simp,
                        height=150,
                        key=f"textarea_{idx}_{chosen_model}"
                    )
                    copy_to_clipboard(final_simp)
                    st.write(f"Temps: {end_simp - start_simp:.1f}s")

                eSimple = {
                    "Alias/Nom": f"{f_name}_{chosen_model}",
                    "Méthode": model_choice,
                    "Modèle": chosen_model,
                    "Durée": human_time(dur_s),
                    "Temps": f"{(end_simp - start_simp):.1f}s",
                    "Coût": "?",
                    "Transcription": final_simp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(eSimple)

            # Nettoyage des segments
            for cfile in chunk_files:
                if os.path.exists(cfile):
                    os.remove(cfile)

        st.write("---")
        st.write("### Aperçu Audio (Historique)")
        for en in st.session_state["history"]:
            st.audio(bytes.fromhex(en["Audio Binaire"]), format="audio/wav")

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
            st.session_state["pwd_attempts"] = st.session_state.get("pwd_attempts",0)+1
            attempts = st.session_state["pwd_attempts"]
            st.error(f"Mot de passe invalide. (Tentative {attempts}/4)")

def main():
    st.set_page_config(page_title="NBL Audio", layout="wide")
    init_state()
    if st.session_state["blocked"]:
        st.error("Vous êtes bloqué suite à trop de tentatives.")
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
