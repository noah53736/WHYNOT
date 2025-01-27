import streamlit as st
import os
import csv
import json
import time
import random
import string
import subprocess
from datetime import datetime
from pydub import AudioSegment, silence

import nova_api

# Fichiers
USER_DATA_CSV = "user_data.csv"
USER_DATA_JSON = "user_data.json"
TRANSCRIPTIONS_JSON = "transcriptions.json"

# set_page_config d'abord
st.set_page_config(page_title="NBL Audio", layout="wide")

# Param. Transformations
MIN_SILENCE_LEN = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
COST_PER_SEC = 0.007

# ============================================================================
# Init Fichiers
# ============================================================================
def init_files():
    if not os.path.exists(USER_DATA_CSV):
        with open(USER_DATA_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["email", "prenom", "nom", "instagram", "date_inscription"])

    if not os.path.exists(USER_DATA_JSON):
        with open(USER_DATA_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)

    if not os.path.exists(TRANSCRIPTIONS_JSON):
        with open(TRANSCRIPTIONS_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)

def load_user_data_json():
    with open(USER_DATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def save_user_data_json(lst):
    with open(USER_DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(lst, f, indent=4)

def load_transcriptions():
    with open(TRANSCRIPTIONS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def save_transcriptions(lst):
    with open(TRANSCRIPTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(lst, f, indent=4)

# ============================================================================
# Utils divers
# ============================================================================
def generate_alias(length=5):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

def human_time(sec: float) -> str:
    s = int(sec)
    if s<60: return f"{s}s"
    elif s<3600:
        m, s2 = divmod(s,60)
        return f"{m}m{s2}s"
    else:
        h,r = divmod(s,3600)
        m, s3 = divmod(r,60)
        return f"{h}h{m}m{s3}s"

def accelerate_ffmpeg(audio_seg: AudioSegment, factor: float)-> AudioSegment:
    if abs(factor-1.0)<1e-2:
        return audio_seg
    tmp_in= "temp_in_acc.wav"
    tmp_out="temp_out_acc.wav"
    audio_seg.export(tmp_in, format="wav")
    remain= factor
    filters=[]
    while remain>2.0:
        filters.append("atempo=2.0")
        remain/=2.0
    while remain<0.5:
        filters.append("atempo=0.5")
        remain/=0.5
    filters.append(f"atempo={remain}")
    f_str=",".join(filters)
    cmd=["ffmpeg","-y","-i",tmp_in,"-filter:a",f_str, tmp_out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seg_new= AudioSegment.from_file(tmp_out, format="wav")
    try:
        os.remove(tmp_in)
        os.remove(tmp_out)
    except: pass
    return seg_new

def remove_silences_classic(audio_seg: AudioSegment,
                            min_silence_len=MIN_SILENCE_LEN,
                            silence_thresh=SIL_THRESH_DB,
                            keep_silence=KEEP_SIL_MS):
    segs= silence.split_on_silence(
        audio_seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    if not segs:
        return audio_seg
    combined= segs[0]
    for s in segs[1:]:
        combined = combined.append(s, crossfade=0)
    return combined

def copy_to_clipboard(text):
    sc= f"""
    <script>
    function copyText(txt){{
        navigator.clipboard.writeText(txt).then(
            ()=>alert('Texte copié dans le presse-papiers!'),
            (err)=>alert('Échec de la copie: '+err)
        );
    }}
    </script>
    <button onclick="copyText(`{text}`)" style="margin-top:5px;">Copier</button>
    """
    st.components.v1.html(sc)

# ============================================================================
# Afficher l'historique de transcriptions
# ============================================================================
def display_transcriptions_history():
    st.sidebar.write("### Historique de Transcriptions")
    hist = load_transcriptions()
    if not hist:
        st.sidebar.info("Aucune transcription enregistrée.")
        return
    # On peut afficher juste un count
    st.sidebar.write(f"{len(hist)} transcriptions enregistrées.")
    # Optionnel : un tableau
    st.sidebar.write("Dernières Transcriptions :")
    for t in hist[::-1][:5]:
        st.sidebar.markdown(f"- **{t.get('user_email','?')}** / {t.get('audio_name','?')} / {t.get('date','?')}")

# ============================================================================
# Formulaire d'identification simple
# ============================================================================
def show_user_form():
    st.subheader("Formulaire d'identification (facultatif)")
    with st.form("user_form"):
        email = st.text_input("Adresse e-mail (optionnel)")
        prenom= st.text_input("Prénom (optionnel)")
        nom= st.text_input("Nom (optionnel)")
        insta= st.text_input("Compte Instagram (optionnel)")
        submitted= st.form_submit_button("Enregistrer & Continuer")
    if submitted:
        # On enregistre
        now_str= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # CSV
        with open(USER_DATA_CSV,"a",newline="",encoding="utf-8") as f:
            w= csv.writer(f)
            w.writerow([email, prenom, nom, insta, now_str])
        # JSON
        userD= load_user_data_json()
        userD.append({
            "email": email,
            "prenom": prenom,
            "nom": nom,
            "instagram": insta,
            "date_inscription": now_str
        })
        with open(USER_DATA_JSON,"w",encoding="utf-8") as f:
            json.dump(userD,f,indent=4)
        st.success("Données enregistrées.")
        st.session_state["user_email"]= email if email else ""

# ============================================================================
# Espace de transcription
# ============================================================================
def show_transcription_space():
    st.title("Espace de Transcription")
    st.markdown("Sélectionnez votre source audio, puis appliquez éventuellement des transformations avant la transcription.")

    # Récup l'email si on veut associer
    user_email = st.session_state.get("user_email","")

    # Transformations
    remove_sil = st.checkbox("Supprimer Silences", False)
    speed_fac = st.slider("Accélération",0.5,4.0,1.0,0.1)

    # Mode double ou simple
    double_mode = st.checkbox("Transcrire simultanément avec deux IA")

    # Import audio
    aud = st.file_uploader("Fichier audio (mp3,wav,m4a,ogg,webm)", type=["mp3","wav","m4a","ogg","webm"])
    if aud:
        st.audio(aud, format=aud.type)
        # Appliquer transformations
        final_seg=None
        try:
            seg= AudioSegment.from_file(aud)
            if remove_sil:
                seg= remove_silences_classic(seg)
            if abs(speed_fac-1.0)>1e-2:
                seg= accelerate_ffmpeg(seg, speed_fac)
            final_seg= seg
            st.success("Audio transformé.")
        except Exception as e:
            st.error(f"Erreur transformations : {e}")

        if final_seg and st.button("Transcrire"):
            try:
                local_temp= f"temp_{aud.name}"
                final_seg.export(local_temp, format="wav")
                startT= time.time()
                if double_mode:
                    # Double => Nova2 + WhisperLarge
                    trans1= nova_api.transcribe_audio(local_temp, st.secrets["NOVA1"], "fr","nova-2")
                    st.info("Première transcription (Nova2) terminée, en attente de la seconde (WhisperLarge).")
                    trans2= nova_api.transcribe_audio(local_temp, st.secrets["NOVA2"], "fr","whisper-large")
                    elapsed= time.time()-startT
                    st.success(f"Double transcription terminée en {elapsed:.1f}s.")
                    # Affichage
                    colL,colR= st.columns(2)
                    with colL:
                        st.subheader("Résultat Nova 2")
                        st.text_area("",trans1,height=180)
                        copy_to_clipboard(trans1)
                    with colR:
                        st.subheader("Résultat WhisperLarge")
                        st.text_area("",trans2,height=180)
                        copy_to_clipboard(trans2)
                    # Enregistrer dans TRANSCRIPTIONS_JSON
                    trans_list= load_transcriptions()
                    nowS= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    trans_list.append({
                        "user_email": user_email,
                        "audio_name": aud.name,
                        "transcript_nova2": trans1,
                        "transcript_whisper": trans2,
                        "double_mode": True,
                        "date": nowS
                    })
                    save_transcriptions(trans_list)
                    st.success("Historique mis à jour.")
                else:
                    # Simple => ex. Nova1
                    trans= nova_api.transcribe_audio(local_temp, st.secrets["NOVA1"], "fr","nova-2")
                    elapsed= time.time()-startT
                    st.success(f"Transcription simple en {elapsed:.1f}s")
                    st.text_area("Texte Final", trans, height=150)
                    copy_to_clipboard(trans)
                    # Enregistrer
                    nowS= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    trans_list= load_transcriptions()
                    trans_list.append({
                        "user_email": user_email,
                        "audio_name": aud.name,
                        "transcript": trans,
                        "double_mode": False,
                        "date": nowS
                    })
                    save_transcriptions(trans_list)
                    st.success("Historique mis à jour.")
                os.remove(local_temp)
            except Exception as e:
                st.error(f"Erreur transcription : {e}")

# ============================================================================
# main
# ============================================================================
def main():
    init_files()
    st.title("NBL Audio – Accueil")
    st.write("Bienvenue sur l'application de transcription. Vous pouvez remplir vos informations ci-dessous (facultatif), ou passer directement à l'espace de transcription.")

    if "user_email" not in st.session_state:
        st.session_state["user_email"] = ""

    # Formulaire
    show_form = st.checkbox("Remplir un Formulaire (facultatif)", False)
    if show_form:
        with st.expander("Formulaire d'Identification"):
            with st.form("user_form"):
                email     = st.text_input("Email (optionnel)")
                prenom    = st.text_input("Prénom (optionnel)")
                nom       = st.text_input("Nom (optionnel)")
                instagram = st.text_input("Instagram (optionnel)")
                sub_form  = st.form_submit_button("Enregistrer")
            if sub_form:
                now_str= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # CSV
                with open(USER_DATA_CSV, "a", newline="", encoding="utf-8") as f:
                    w= csv.writer(f)
                    w.writerow([email, prenom, nom, instagram, now_str])
                # JSON
                j_data= load_user_data_json()
                j_data.append({
                    "email": email,
                    "prenom": prenom,
                    "nom": nom,
                    "instagram": instagram,
                    "date_inscription": now_str
                })
                with open(USER_DATA_JSON,"w",encoding="utf-8") as f:
                    json.dump(j_data, f, indent=4)
                st.success("Données enregistrées.")
                st.session_state["user_email"]= email

    # Espace de transcription
    st.write("---")
    show_transcription_space()

    # Historique des transcriptions dans la sidebar
    st.sidebar.write("---")
    st.sidebar.write("## Historique")
    trans_hist= load_transcriptions()
    if not trans_hist:
        st.sidebar.info("Aucune transcription pour l'instant.")
    else:
        st.sidebar.write(f"{len(trans_hist)} transcription(s) enregistrée(s).")
        for tx in trans_hist[::-1][:5]:
            st.sidebar.markdown(f"- {tx.get('audio_name','?')} / {tx.get('date','?')}")

if __name__=="__main__":
    main()
