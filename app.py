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
    """ Récupère les clés DeepGram (NOVA...). """
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def pick_key(keys):
    """ On ne fait pas de rotation multiple. On prend la clé courante. """
    if not keys:
        return None
    idx = st.session_state["dg_key_index"]
    if idx>=len(keys):
        st.session_state["dg_key_index"] = 0
        idx=0
    return keys[idx]

def generate_alias(length=5):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

def human_time(sec: float) -> str:
    sec = int(sec)
    if sec<60:
        return f"{sec}s"
    elif sec<3600:
        m, s = divmod(sec,60)
        return f"{m}m{s}s"
    else:
        h, r = divmod(sec,3600)
        m, s = divmod(r,60)
        return f"{h}h{m}m{s}s"

def copy_button(text):
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
    <button onclick="copyText_{uid}()" style="font-size:15px; margin:5px; padding:5px;">COPIER</button>
    """
    st.components.v1.html(script_html)

def password_gate():
    st.title("Accès Protégé (Mot de Passe)")
    code_in = st.text_input("Code (4 chiffres) :", "", max_chars=4, type="password")
    if st.button("Valider"):
        if st.session_state.get("blocked", False):
            st.warning("Vous êtes déjà bloqué.")
            st.stop()
        if st.session_state.get("pwd_attempts",0)>=4:
            st.error("Trop de tentatives. Session bloquée.")
            st.session_state["blocked"] = True
            st.stop()

        real_pwd = st.secrets.get("APP_PWD","1234")
        if code_in == real_pwd:
            st.session_state["authorized"] = True
            st.success("Accès autorisé.")
        else:
            st.session_state["pwd_attempts"] +=1
            st.error(f"Mot de passe invalide. (Tentative {st.session_state['pwd_attempts']}/4)")

def chunk_if_needed(raw_data: bytes, threshold=25*1024*1024):
    """ Segmente en blocs ~25MB, usage si WhisperLarge & audio>25MB. """
    if len(raw_data)<=threshold:
        return [AudioSegment.from_file(io.BytesIO(raw_data))]
    st.info("Fichier volumineux => segmentation 25MB pour Whisper Large.")
    segf = AudioSegment.from_file(io.BytesIO(raw_data))
    segments=[]
    length_ms = len(segf)
    step_ms = int(length_ms*(threshold/len(raw_data)))
    start=0
    while start<length_ms:
        end = min(start+step_ms,length_ms)
        segments.append(segf[start:end])
        start=end
    return segments

def main_app():
    st.title("NBL Audio")
    st.markdown("""
**Whisper Large** : Technologie IA très précise (~99%), peut prendre 5-10 minutes sur des audios de plus d'1h.
Ne fermez pas la page pendant la transcription.
""")

    # micro par defaut => index=1
    src = st.radio("Source Audio", ["Fichier (Upload)","Micro (Enregistrement)"], index=1)
    audio_data = []
    file_names = []

    if src=="Fichier (Upload)":
        upfiles = st.file_uploader("Fichiers audio multiples", 
            type=["mp3","wav","m4a","ogg","webm"],
            accept_multiple_files=True
        )
        if upfiles:
            for f in upfiles:
                audio_data.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        nm = st.number_input("Nb micros",1,4,1)
        for i in range(nm):
            mic_input = st.audio_input(f"Micro {i+1}")
            if mic_input:
                audio_data.append(mic_input.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic_input, format=mic_input.type)

    st.write("---")
    st.subheader("Double Transcription (par défaut) => Nova2 (rapide) puis Whisper (précis)")
    double_trans = st.checkbox("Double Transcription", value=True)

    c1,c2= st.columns([1,1])
    with c1:
        model_simple = st.selectbox("Modèle unique (si double off)", ["Nova 2","Whisper Large"])
    with c2:
        lang_main = "fr"
        if model_simple=="Whisper Large":
            lang_main = st.selectbox("Langue (Whisper)?", ["fr","en"])

    if st.button("Transcrire") and audio_data:
        st.info("Traitement en cours, merci de patienter...")
        keys = get_dg_keys()
        if not keys:
            st.error("Aucune clé DeepGram dispo.")
            return
        for idx, raw in enumerate(audio_data):
            segA = AudioSegment.from_file(io.BytesIO(raw))
            dur_s = len(segA)/1000.0
            fname = file_names[idx] if idx<len(file_names) else f"Audio_{idx+1}"
            st.write(f"### Fichier {idx+1}: {fname} (Durée: {human_time(dur_s)})")

            # Export en un bloc
            tmpA = f"temp_input_{idx}.wav"
            segA.export(tmpA, format="wav")

            if double_trans:
                # 1) Nova 2
                st.info(f"Nova 2 (rapide) pour '{fname}'...")
                start_nova=time.time()
                keyN = pick_key(keys)
                txt_nova = nova_api.transcribe_audio(tmpA, keyN, "fr", "nova-2")
                end_nova=time.time()
                st.success(f"Nova 2 terminé pour '{fname}'. Affichage ci-dessous.")
                colL, colR = st.columns(2)
                with colL:
                    st.subheader(f"NOVA 2 - {fname}")
                    st.text_area(f"Nova_{idx}", txt_nova, height=120)
                    copy_button(txt_nova)
                    st.write(f"Nova2 => ~{end_nova - start_nova:.1f}s")

                # 2) Whisper Large
                st.info("Lancement Whisper Large (technologie IA précise, 5-10 min possible pour 1h30).")
                # segment?
                doSeg = (len(raw)>25*1024*1024)
                partial_whisp = []
                start_whisp=time.time()
                with colR:
                    st.subheader(f"WHISPER LARGE - {fname}")
                    # On va remplir au fur et à mesure
                    whisp_container = st.empty()
                if doSeg:
                    segs = chunk_if_needed(raw, threshold=25*1024*1024)
                    for s_i, s_sub in enumerate(segs):
                        wout = f"temp_whisp_{idx}_{s_i}.wav"
                        s_sub.export(wout, format="wav")
                        kw = pick_key(keys)
                        piece = nova_api.transcribe_audio(wout, kw, lang_main, "whisper-large")
                        partial_whisp.append(piece)
                        current_text = " ".join(partial_whisp)
                        with colR:
                            whisp_container.text_area(
                                f"Whisper_{idx}",
                                current_text,
                                height=120
                            )
                            copy_button(current_text)
                        os.remove(wout)
                else:
                    kw2 = pick_key(keys)
                    one_shot = nova_api.transcribe_audio(tmpA, kw2, lang_main, "whisper-large")
                    partial_whisp.append(one_shot)
                    with colR:
                        whisp_container.text_area(
                            f"Whisper_{idx}",
                            one_shot,
                            height=120
                        )
                        copy_button(one_shot)
                end_whisp=time.time()
                st.success(f"Whisper Large fini pour '{fname}' => {(end_whisp - start_whisp):.1f}s")

            else:
                # Mode simple
                if model_simple=="Nova 2":
                    cmodel="nova-2"
                    clang="fr"
                else:
                    cmodel="whisper-large"
                    clang=lang_main
                st.info(f"Transcription {model_simple} => potentiel 5-10min pour un gros audio.")
                doSeg2 = (cmodel=="whisper-large" and len(raw)>25*1024*1024)
                partialS=[]
                startS=time.time()
                cA,_ = st.columns([1,1])
                with cA:
                    st.subheader(f"{model_simple} - {fname}")
                    txt_box = st.empty()
                if doSeg2:
                    # segmentation
                    segs2=chunk_if_needed(raw, threshold=25*1024*1024)
                    for s_j, subseg2 in enumerate(segs2):
                        wv2=f"tempsimp_{idx}_{s_j}.wav"
                        subseg2.export(wv2, format="wav")
                        keyS=pick_key(keys)
                        piece2=nova_api.transcribe_audio(wv2, keyS, clang, cmodel)
                        partialS.append(piece2)
                        combined=" ".join(partialS)
                        with cA:
                            txt_box.text_area(
                                f"WhispSimple_{idx}",
                                combined,
                                height=120
                            )
                            copy_button(combined)
                        os.remove(wv2)
                else:
                    keyS2=pick_key(keys)
                    finalSimple=nova_api.transcribe_audio(tmpA, keyS2, clang, cmodel)
                    partialS.append(finalSimple)
                    with cA:
                        txt_box.text_area(
                            f"Simple_{idx}",
                            finalSimple,
                            height=120
                        )
                        copy_button(finalSimple)
                endS=time.time()
                st.success(f"{model_simple} terminé pour '{fname}' => ~{endS - startS:.1f}s")

            if os.path.exists(tmpA):
                os.remove(tmpA)

def main():
    try:
        if "init" not in st.session_state:
            init_state()
            st.session_state["init"]=True
        if st.session_state.get("blocked",False):
            st.error("Session bloquée suite à tentatives.")
            st.stop()
        if not st.session_state.get("authorized",False):
            password_gate()
            if st.session_state.get("authorized",False):
                main_app()
        else:
            main_app()
    except Exception as e:
        st.error(f"Erreur Principale : {e}")
        traceback.print_exc()

if __name__=="__main__":
    main()
