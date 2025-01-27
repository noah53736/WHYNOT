import streamlit as st
import os
import json
import random
import string
import time
import subprocess
from datetime import datetime
from pydub import AudioSegment, silence

import nova_api

# set_page_config : Doit être le 1er appel Streamlit
st.set_page_config(page_title="NBL Audio", layout="wide")

TRANSCRIPTIONS_JSON = "transcriptions.json"

# Paramètres de transformations
MIN_SILENCE_LEN = 700
SIL_THRESH_DB    = -35
KEEP_SIL_MS      = 50
COST_PER_SEC     = 0.007

# ============================================================================
# Fonctions de gestion des transcriptions
# ============================================================================
def init_transcriptions():
    if not os.path.exists(TRANSCRIPTIONS_JSON):
        with open(TRANSCRIPTIONS_JSON,"w",encoding="utf-8") as f:
            json.dump([],f,indent=4)

def load_transcriptions():
    with open(TRANSCRIPTIONS_JSON,"r",encoding="utf-8") as f:
        return json.load(f)

def save_transcriptions(lst):
    with open(TRANSCRIPTIONS_JSON,"w",encoding="utf-8") as f:
        json.dump(lst,f,indent=4)

def generate_alias(length=5):
    return "".join(random.choices(string.ascii_uppercase+string.digits, k=length))

def human_time(sec: float)-> str:
    s = int(sec)
    if s<60:
        return f"{s}s"
    elif s<3600:
        m, s2= divmod(s,60)
        return f"{m}m{s2}s"
    else:
        h, r= divmod(s,3600)
        m, s3= divmod(r,60)
        return f"{h}h{m}m{s3}s"

# ============================================================================
# Transformations
# ============================================================================
def accelerate_ffmpeg(seg: AudioSegment, factor: float)-> AudioSegment:
    if abs(factor-1.0)<1e-2:
        return seg
    tmp_in="temp_in_acc.wav"
    tmp_out="temp_out_acc.wav"
    seg.export(tmp_in, format="wav")
    remain=factor
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
    new_seg= AudioSegment.from_file(tmp_out, format="wav")
    try:
        os.remove(tmp_in)
        os.remove(tmp_out)
    except: pass
    return new_seg

def remove_silences_classic(seg: AudioSegment,
                            min_silence_len=MIN_SILENCE_LEN,
                            silence_thresh=SIL_THRESH_DB,
                            keep_silence=KEEP_SIL_MS):
    parts= silence.split_on_silence(
        seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    if not parts:
        return seg
    comb= parts[0]
    for p in parts[1:]:
        comb= comb.append(p, crossfade=0)
    return comb

def copy_to_clipboard(text):
    # Petit script HTML/JS
    sc= f"""
    <script>
    function copyText(txt){{
        navigator.clipboard.writeText(txt).then(
            ()=>alert('Texte copié!'),
            (err)=>alert('Échec copie : '+err)
        );
    }}
    </script>
    <button onclick="copyText(`{text}`)" style="margin-top:5px;">Copier</button>
    """
    st.components.v1.html(sc)

def display_side_history():
    st.sidebar.header("Historique")
    st.sidebar.write("---")
    hist= load_transcriptions()
    if not hist:
        st.sidebar.info("Aucune transcription.")
    else:
        st.sidebar.write(f"{len(hist)} au total.")
        for t in hist[::-1][:6]:
            st.sidebar.markdown(f"- {t.get('audio_name','?')} / {t.get('date','?')}")

# ============================================================================
# main
# ============================================================================
def main():
    init_transcriptions()
    display_side_history()

    st.title("NBL Audio : Transcription Grand Public")

    # Barre latérale
    st.sidebar.header("Options")
    remove_sil = st.sidebar.checkbox("Supprimer Silences",False)
    speed_factor= st.sidebar.slider("Accélération Audio",0.5,4.0,1.0,0.1)
    double_mode= st.sidebar.checkbox("Transcrire Simultanément (Nova2 + Whisper)",False)
    st.sidebar.write("---")
    st.sidebar.header("Modèle (si simple)")
    model_choice= st.sidebar.radio("Modèle IA:", ["Nova 2","Whisper Large"])
    model_map= {"Nova 2":"nova-2","Whisper Large":"whisper-large"}
    st.sidebar.write("---")
    st.sidebar.header("Langue")
    lang_choice= st.sidebar.radio("Langue:", ["fr","en"])

    # Choix du type d'entrée
    st.write("## Choisissez le Mode d'Entrée")
    mode_in= st.radio("", ["Fichier (Upload)","Micro (Enregistrement)","Multi (Plusieurs)"])

    sources=[]
    if mode_in=="Fichier (Upload)":
        upf= st.file_uploader("Fichier audio (mp3,wav,m4a,ogg,webm)", type=["mp3","wav","m4a","ogg","webm"])
        if upf:
            st.audio(upf, format=upf.type)
            rename= st.text_input("Nom (optionnel)", upf.name)
            sources.append((upf, rename))

    elif mode_in=="Micro (Enregistrement)":
        mic= st.audio_input("Enregistrer micro")
        if mic:
            st.audio(mic, format=mic.type)
            rename= st.text_input("Nom (optionnel)", "micro.wav")
            sources.append((mic, rename))

    else:
        st.write("Vous pouvez charger jusqu'à 5 segments.")
        for i in range(5):
            colA, colB= st.columns([2,1])
            with colA:
                fu= st.file_uploader(f"Fichier #{i+1}", type=["mp3","wav","m4a","ogg","webm"], key=f"f_{i}")
                if fu:
                    st.audio(fu, format=fu.type)
                    nm= st.text_input(f"Nom #{i+1}", fu.name, key=f"nm_{i}")
                    sources.append((fu,nm))
            with colB:
                mic2= st.audio_input(f"Micro #{i+1}", key=f"mic_{i}")
                if mic2:
                    st.audio(mic2, format=mic2.type)
                    nm2= st.text_input(f"Nom Micro #{i+1}", "micro.wav", key=f"nm2_{i}")
                    sources.append((mic2,nm2))

    final_segments=[]

    # Bouton transformations
    if len(sources)>0 and st.button("Appliquer Transformations"):
        for idx, (srcObj, rename) in enumerate(sources):
            try:
                local_tmp= f"temp_in_{idx}.wav"
                with open(local_tmp,"wb") as ff:
                    ff.write(srcObj.read())
                seg= AudioSegment.from_file(local_tmp)
                if remove_sil:
                    seg= remove_silences_classic(seg)
                if abs(speed_factor-1.0)>1e-2:
                    seg= accelerate_ffmpeg(seg, speed_factor)
                final_segments.append((seg, rename))
                st.success(f"Segment #{idx+1} transformé: {rename}")
                os.remove(local_tmp)
            except Exception as e:
                st.error(f"Erreur segment #{idx+1}: {e}")

    # Bouton Transcrire
    if len(final_segments)>0 and st.button("Transcrire Maintenant"):
        trans_list= load_transcriptions()
        startT= time.time()
        for idx, (seg, nm) in enumerate(final_segments):
            st.write(f"### Segment #{idx+1}: {nm}")
            path_loc= f"trans_segment_{idx}.wav"
            seg.export(path_loc, format="wav")
            if double_mode:
                # 2 transcriptions
                st.info("Nova2 en cours...")
                txt1= nova_api.transcribe_audio(path_loc, st.secrets["NOVA1"], language=lang_choice, model_name="nova-2")
                st.info("WhisperLarge en cours...")
                txt2= nova_api.transcribe_audio(path_loc, st.secrets["NOVA2"], language=lang_choice, model_name="whisper-large")
                colA,colB= st.columns(2)
                with colA:
                    st.subheader("Nova2")
                    st.text_area("",txt1,height=150)
                    copy_to_clipboard(txt1)
                with colB:
                    st.subheader("WhisperLarge")
                    st.text_area("",txt2,height=150)
                    copy_to_clipboard(txt2)
                trans_list.append({
                    "audio_name": nm,
                    "double": True,
                    "transcript_nova2": txt1,
                    "transcript_whisper": txt2,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                # Single
                if model_choice=="Nova 2":
                    keyUsed= st.secrets["NOVA1"]
                    modelUsed= "nova-2"
                else:
                    keyUsed= st.secrets["NOVA2"]
                    modelUsed= "whisper-large"
                st.info(f"Transcription {model_choice} en cours...")
                txt= nova_api.transcribe_audio(path_loc, keyUsed, lang_choice, modelUsed)
                st.text_area("",txt,height=150)
                copy_to_clipboard(txt)
                trans_list.append({
                    "audio_name": nm,
                    "double": False,
                    "model": modelUsed,
                    "transcript": txt,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            os.remove(path_loc)
        save_transcriptions(trans_list)
        elapsed= time.time()-startT
        st.success(f"Toutes les transcriptions terminées en {elapsed:.1f}s.")
        st.experimental_rerun()

if __name__=="__main__":
    main()
