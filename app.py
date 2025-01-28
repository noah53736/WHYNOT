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
    # Agrandir un peu le bouton, et pointer sur la div correspondante
    script_html = f"""
    <script>
        function copyText() {{
            navigator.clipboard.writeText(`{text}`).then(function() {{
                alert('Texte Whisper/Transcrit copié dans le presse-papiers!');
            }}, function(err) {{
                alert('Échec de la copie : ' + err);
            }});
        }}
    </script>
    <button onclick="copyText()" style="margin:5px; font-size:16px; padding:6px;">COPIER</button>
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
                "Temps": wh["Temps"]
            })
        st.sidebar.table(table[::-1])
        if st.sidebar.button("Vider Historique Whisper", key="clear_whisp"):
            st.session_state["whisper_history"].clear()
            st.experimental_rerun()
    else:
        st.sidebar.info("Aucun historique Whisper.")

#########################
#   SEGMENTATION 25MB   #
#########################
def segment_if_needed(raw_data: bytes, threshold=25*1024*1024):
    """
    Découpe en blocs de 25MB si c'est plus grand,
    UNIQUEMENT pour Whisper Large usage.
    """
    if len(raw_data) <= threshold:
        return [AudioSegment.from_file(io.BytesIO(raw_data))]
    st.info("Fichier volumineux => segmentation par blocs de 25MB (Whisper Large).")
    full_seg = AudioSegment.from_file(io.BytesIO(raw_data))
    segments = []
    length_ms = len(full_seg)
    # proportionnel
    step_ms = int(length_ms*(threshold/len(raw_data)))
    start = 0
    while start < length_ms:
        end = min(start+step_ms, length_ms)
        segments.append(full_seg[start:end])
        start = end
    return segments

#########################
#    APP PRINCIPALE     #
#########################
def app_main():
    display_whisper_history()

    st.title("NBL Audio")

    st.markdown("""
**Informations sur les IA :**
- **Nova 2** : IA rapide, résultat en quelques secondes.  
- **Whisper Large** : IA plus lente (2-5 min pour des audios de 1h30), très précise (~99%).  
**Ne quittez pas cette page Web** pendant la transcription, surtout pour de gros fichiers.
    """)

    src_type = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"], index=1)
    audio_data_list = []
    file_names = []

    if src_type=="Fichier (Upload)":
        upfs = st.file_uploader(
            "Importer vos fichiers audio (jusqu'à plusieurs fichiers)",
            type=["mp3","wav","m4a","ogg","webm"],
            accept_multiple_files=True
        )
        if upfs:
            for f in upfs:
                audio_data_list.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        nb = st.number_input("Nombre de micros", 1,4,1)
        for i in range(nb):
            mic_inp = st.audio_input(f"Micro {i+1}")
            if mic_inp:
                audio_data_list.append(mic_inp.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic_inp, format=mic_inp.type)

    st.write("---")
    st.subheader("Choix de Transcription")
    st.markdown("""
Double Transcription (activée par défaut) :  
- Lance d'abord Nova 2 (résultat rapide)  
- Puis **Whisper Large** (l'IA la plus précise, patience nécessaire).
    """)
    double_trans = st.checkbox("Double Transcription (Nova2 -> Whisper)", value=True)

    colA, colB = st.columns([1,1])
    with colA:
        single_model = st.selectbox("Modèle (si double non coché)", ["Nova 2","Whisper Large"])
    with colB:
        lang_main = "fr"
        if single_model=="Whisper Large":
            lang_main = st.selectbox("Langue (Whisper)", ["fr","en"])

    if st.button("Transcrire") and audio_data_list:
        st.info("Début de la transcription... Merci de patienter.")
        keys = get_dg_keys()
        if not keys:
            st.error("Aucune clé DeepGram configurée.")
            st.stop()

        for idx, raw in enumerate(audio_data_list):
            seg = AudioSegment.from_file(io.BytesIO(raw))
            dur_s = len(seg)/1000.0
            f_name = file_names[idx] if idx<len(file_names) else f"Fichier_{idx+1}"
            st.write(f"### Fichier {idx+1}: {f_name} (Durée: {human_time(dur_s)})")

            temp_main = f"temp_main_{idx}.wav"
            seg.export(temp_main, format="wav")

            if double_trans:
                # 1) Nova 2
                st.write(f"**{f_name} => Nova 2** (rapide)")
                start_nova = time.time()
                k_nova = pick_key(keys)
                txt_nova = nova_api.transcribe_audio(temp_main, k_nova, "fr", "nova-2")
                end_nova = time.time()
                st.success(f"Nova 2 terminé pour '{f_name}'. Résultat ci-dessous. Lancement Whisper Large...")

                # 2) Whisper Large
                st.info("Veuillez patienter pour Whisper Large (IA plus lente, très précise).")
                # Si >25MB => segmentation
                do_seg = (os.path.getsize(temp_main)>25*1024*1024)
                if do_seg:
                    st.info("Fichier >25MB => segmentation en blocs (Whisper).")
                    rawsize = os.path.getsize(temp_main)
                    segm = segment_if_needed(raw, threshold=25*1024*1024)
                    # export segments
                    cfiles = []
                    for si, s_sub in enumerate(segm):
                        wout = f"whisp_seg_{idx}_{si}.wav"
                        s_sub.export(wout, format="wav")
                        cfiles.append(wout)
                    start_whisp = time.time()
                    txtw_list=[]
                    for cf in cfiles:
                        k_whisp = pick_key(keys)
                        tw = nova_api.transcribe_audio(cf, k_whisp, lang_main, "whisper-large")
                        txtw_list.append(tw)
                    end_whisp = time.time()
                    for cf2 in cfiles:
                        os.remove(cf2)
                    txt_whisp = " ".join(txtw_list)
                else:
                    start_whisp = time.time()
                    k_whisp = pick_key(keys)
                    txt_whisp = nova_api.transcribe_audio(temp_main, k_whisp, lang_main, "whisper-large")
                    end_whisp = time.time()

                cA, cB = st.columns(2)
                with cA:
                    st.subheader(f"NOVA 2 - {f_name}")
                    st.text_area(
                        f"Nova2_{idx}",
                        txt_nova,
                        height=130
                    )
                    copy_to_clipboard(txt_nova)
                    st.write(f"Temps Nova2: {end_nova - start_nova:.1f}s")

                with cB:
                    st.subheader(f"WHISPER LARGE - {f_name}")
                    st.text_area(
                        f"Whisp_{idx}",
                        txt_whisp,
                        height=130
                    )
                    copy_to_clipboard(txt_whisp)
                    st.write(f"Temps Whisper: {time.time()-start_whisp:.1f}s")

                # Stocker histo whisper
                eWhisp = {
                    "Alias/Nom": f"{f_name}_Whisper",
                    "Durée": f"{human_time(dur_s)}",
                    "Temps": f"{(time.time()-start_whisp):.1f}s",
                    "Transcription": txt_whisp
                }
                st.session_state["whisper_history"].append(eWhisp)

            else:
                # Simple
                if single_model=="Nova 2":
                    chosen_model="nova-2"
                    chosen_lang="fr"
                else:
                    chosen_model="whisper-large"
                    chosen_lang=lang_main

                st.info(f"Lancement de {single_model} pour '{f_name}'...")
                start_simp = time.time()
                # si whisp>25MB => seg
                do_seg2 = (chosen_model=="whisper-large" and os.path.getsize(temp_main)>25*1024*1024)
                if do_seg2:
                    st.info("Fichier volumineux => segmentation (Whisper).")
                    segm2 = segment_if_needed(raw, 25*1024*1024)
                    cfs2=[]
                    for si2, ssub2 in enumerate(segm2):
                        tout2=f"whisp2_seg_{idx}_{si2}.wav"
                        ssub2.export(tout2, format="wav")
                        cfs2.append(tout2)
                    txtlist2=[]
                    for cf3 in cfs2:
                        ksim = pick_key(keys)
                        tx_ = nova_api.transcribe_audio(cf3, ksim, chosen_lang, chosen_model)
                        txtlist2.append(tx_)
                    for cf4 in cfs2:
                        os.remove(cf4)
                    final_simp = " ".join(txtlist2)
                    end_simp = time.time()
                else:
                    ksim = pick_key(keys)
                    final_simp = nova_api.transcribe_audio(temp_main, ksim, chosen_lang, chosen_model)
                    end_simp = time.time()

                st.success(f"Transcription terminée pour '{f_name}'.")
                cA,_ = st.columns([1,1])
                with cA:
                    st.subheader(f"{single_model} - {f_name}")
                    st.text_area(
                        f"Single_{idx}",
                        final_simp,
                        height=130
                    )
                    copy_to_clipboard(final_simp)
                    st.write(f"Temps: {end_simp - start_simp:.1f}s")

                # Si c'est whisper => log histo
                if chosen_model=="whisper-large":
                    eW2 = {
                        "Alias/Nom": f"{f_name}_Whisper",
                        "Durée": f"{human_time(dur_s)}",
                        "Temps": f"{(end_simp - start_simp):.1f}s",
                        "Transcription": final_simp
                    }
                    st.session_state["whisper_history"].append(eW2)

            if os.path.exists(temp_main):
                os.remove(temp_main)

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
