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
import traceback

#########################
#   CONFIG PAGE UNIQUE  #
#########################
st.set_page_config(page_title="NBL Audio", layout="wide")

#########################
#   INITIALISATIONS     #
#########################
def init_state():
    if "dg_key_index" not in st.session_state:
        st.session_state["dg_key_index"] = 0
    if "pwd_attempts" not in st.session_state:
        st.session_state["pwd_attempts"] = 0
    if "authorized" not in st.session_state:
        st.session_state["authorized"] = False
    if "blocked" not in st.session_state:
        st.session_state["blocked"] = False

def get_dg_keys():
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def pick_key(keys):
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

def copy_button_js(text):
    # Bouton plus grand, message confirmation
    # ID unique ?
    uid = generate_alias(6)
    script_html = f"""
    <script>
        function copyText_{uid}() {{
            navigator.clipboard.writeText(`{text}`).then(function() {{
                alert('Texte copié dans le presse-papiers!');
            }}, function(err) {{
                alert('Échec de la copie : ' + err);
            }});
        }}
    </script>
    <button onclick="copyText_{uid}()" style="font-size:16px; padding:5px; margin:5px;">COPIER</button>
    """
    st.components.v1.html(script_html)

def password_gate():
    st.title("Accès Protégé (Mot de Passe)")
    code_in = st.text_input("Code 4 chiffres:", value="", max_chars=4, type="password")
    if st.button("Valider"):
        if st.session_state.get("blocked", False):
            st.warning("Vous êtes déjà bloqué.")
            st.stop()
        if st.session_state.get("pwd_attempts", 0) >= 4:
            st.error("Trop de tentatives. Session bloquée.")
            st.session_state["blocked"] = True
            st.stop()

        real_pwd = st.secrets.get("APP_PWD","1234")
        if code_in==real_pwd:
            st.session_state["authorized"] = True
            st.success("Accès autorisé.")
        else:
            st.session_state["pwd_attempts"] = st.session_state.get("pwd_attempts",0)+1
            st.error(f"Mot de passe erroné. Tentative {st.session_state['pwd_attempts']}/4.")

def segment_if_needed(raw_data: bytes, threshold=25*1024*1024):
    if len(raw_data) <= threshold:
        return [AudioSegment.from_file(io.BytesIO(raw_data))]
    st.info("Fichier volumineux => segmentation 25MB (Whisper).")
    seg_full = AudioSegment.from_file(io.BytesIO(raw_data))
    segments = []
    length_ms = len(seg_full)
    step_ms = int(length_ms*(threshold/len(raw_data))) # proportionnel
    start = 0
    while start<length_ms:
        end = min(start+step_ms, length_ms)
        segments.append(seg_full[start:end])
        start=end
    return segments

def main_app():
    st.title("NBL Audio")
    st.markdown("""
**Nova 2** : IA rapide  
**Whisper Large** : IA plus lente (2-5 minutes pour 1h30), plus précise (~99%).  
_Ne quittez pas cette page_ durant le processus.
""")

    # Micro par défaut => index=1
    src = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"], index=1)
    audio_data = []
    file_names = []

    if src=="Fichier (Upload)":
        upf = st.file_uploader("Fichiers audio multiples possible", 
            type=["mp3","wav","m4a","ogg","webm"],
            accept_multiple_files=True
        )
        if upf:
            for f in upf:
                audio_data.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        nb = st.number_input("Nb micros", 1,4,1)
        for i in range(nb):
            mic_in = st.audio_input(f"Micro {i+1}")
            if mic_in:
                audio_data.append(mic_in.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic_in, format=mic_in.type)

    st.write("---")
    st.subheader("Double Transcription (Activée par défaut) : Nova2 → Whisper")
    double_trans = st.checkbox("Double Transcription", value=True)

    col1, col2 = st.columns([1,1])
    with col1:
        model_main = st.selectbox("Modèle unique (si non double)", ["Nova 2","Whisper Large"])
    with col2:
        lang_main = "fr"
        if model_main=="Whisper Large":
            lang_main = st.selectbox("Langue (Whisper)?", ["fr","en"])

    if st.button("Transcrire") and audio_data:
        st.info("Démarrage de la transcription, patientez s'il vous plaît...")
        keys = get_dg_keys()
        if not keys:
            st.error("Aucune clé DeepGram disponible.")
            return

        for idx, raw in enumerate(audio_data):
            segA = AudioSegment.from_file(io.BytesIO(raw))
            dur_s = len(segA)/1000.0
            name_ = file_names[idx] if idx<len(file_names) else f"F_{idx+1}"

            st.write(f"### Fichier {idx+1}: {name_} (Durée {human_time(dur_s)})")

            # On export en un bloc
            tmpA = f"temp_main_{idx}.wav"
            segA.export(tmpA, format="wav")

            if double_trans:
                # 1) Nova 2
                st.info(f"Transcription Nova 2 (rapide) pour '{name_}'...")
                start_nova = time.time()
                kN = pick_key(keys)
                txt_nova = nova_api.transcribe_audio(tmpA, kN, "fr", "nova-2")
                end_nova = time.time()
                st.success(f"Nova 2 terminé pour '{name_}'. Résultat ci-dessous.")
                # On l'affiche direct
                cL, cR = st.columns(2)
                with cL:
                    st.subheader(f"NOVA 2 - {name_}")
                    st.text_area(f"Nova_{idx}", txt_nova, height=130)
                    copy_button_js(txt_nova)

                # 2) Whisper
                st.info(f"Début Whisper Large (plus précise) pour '{name_}'.")
                # seg si >25MB
                do_seg = (len(raw)>25*1024*1024)
                final_whisp = ""
                whisp_box = cR.empty()  # un conteneur
                start_whisp = time.time()

                if do_seg:
                    # seg
                    st.info("Fichier >25MB => segmentation.")
                    segs = segment_if_needed(raw, threshold=25*1024*1024)
                    partial = []
                    for s_i, ssub in enumerate(segs):
                        wavX = f"temp_whisp_{idx}_{s_i}.wav"
                        ssub.export(wavX, format="wav")
                        kW = pick_key(keys)
                        # On transcrit => on concat
                        piece = nova_api.transcribe_audio(wavX, kW, lang_main, "whisper-large")
                        partial.append(piece)
                        final_whisp = " ".join(partial)  
                        # On met à jour l'affichage en direct
                        with cR:
                            st.subheader(f"WHISPER LARGE - {name_}")
                            st.text_area(
                                f"whispPartial_{idx}",
                                final_whisp,
                                height=130
                            )
                            copy_button_js(final_whisp)
                        os.remove(wavX)
                else:
                    # one shot
                    kW2 = pick_key(keys)
                    partial_txt = nova_api.transcribe_audio(tmpA, kW2, lang_main, "whisper-large")
                    final_whisp = partial_txt
                    with cR:
                        st.subheader(f"WHISPER LARGE - {name_}")
                        st.text_area(
                            f"whispPartial_{idx}",
                            final_whisp,
                            height=130
                        )
                        copy_button_js(final_whisp)

                end_whisp = time.time()
                st.success(f"Whisper Large terminé pour '{name_}'.")

            else:
                # single
                if model_main=="Nova 2":
                    chosen_model="nova-2"
                    chosen_lang="fr"
                else:
                    chosen_model="whisper-large"
                    chosen_lang=lang_main
                st.info(f"Transcription {model_main} pour '{name_}' (patientez).")
                do_seg2 = (chosen_model=="whisper-large" and len(raw)>25*1024*1024)
                final_s=""
                cA, _ = st.columns([1,1])
                if do_seg2:
                    st.info("Segmentation car >25MB, Whisper Large.")
                    segs2 = segment_if_needed(raw, threshold=25*1024*1024)
                    partial2=[]
                    for s_i2, ssub2 in enumerate(segs2):
                        wav2 = f"temp_whisp2_{idx}_{s_i2}.wav"
                        ssub2.export(wav2, format="wav")
                        key_ = pick_key(keys)
                        pc = nova_api.transcribe_audio(wav2, key_, chosen_lang, chosen_model)
                        partial2.append(pc)
                        final_s = " ".join(partial2)
                        with cA:
                            st.subheader(f"{model_main} - {name_}")
                            st.text_area(
                                f"whispSeg_{idx}",
                                final_s,
                                height=130
                            )
                            copy_button_js(final_s)
                        os.remove(wav2)
                else:
                    key3 = pick_key(keys)
                    final_s = nova_api.transcribe_audio(tmpA, key3, chosen_lang, chosen_model)
                    with cA:
                        st.subheader(f"{model_main} - {name_}")
                        st.text_area(
                            f"Res_{idx}",
                            final_s,
                            height=130
                        )
                        copy_button_js(final_s)

            if os.path.exists(tmpA):
                os.remove(tmpA)

def main():
    if "init" not in st.session_state:
        init_state()
        st.session_state["init"] = True

    if st.session_state.get("blocked",False):
        st.error("Session bloquée (trop de tentatives).")
        st.stop()
    if not st.session_state.get("authorized",False):
        password_gate()
        if st.session_state.get("authorized",False):
            main_app()
    else:
        main_app()

def main_wrapper():
    try:
        main()
    except Exception as e:
        st.error(f"Erreur Principale : {e}")
        traceback.print_exc()

if __name__=="__main__":
    main_wrapper()
