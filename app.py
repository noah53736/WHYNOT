import streamlit as st
import os
import json
import time
import random
import string
import subprocess
from datetime import datetime
from pydub import AudioSegment, silence

import nova_api

# set_page_config : au tout début
st.set_page_config(page_title="NBL Audio", layout="wide")

TRANSCRIPTIONS_JSON = "transcriptions.json"

MIN_SILENCE_LEN = 700
SIL_THRESH_DB    = -35
KEEP_SIL_MS      = 50
COST_PER_SEC     = 0.007

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
    s= int(sec)
    if s<60:
        return f"{s}s"
    elif s<3600:
        m, s2= divmod(s,60)
        return f"{m}m{s2}s"
    else:
        h, r= divmod(s,3600)
        m, s3= divmod(r,60)
        return f"{h}h{m}m{s3}s"

def accelerate_ffmpeg(seg: AudioSegment, factor: float)->AudioSegment:
    if abs(factor - 1.0)<1e-2:
        return seg
    tmp_in= "temp_in_acc.wav"
    tmp_out="temp_out_acc.wav"
    seg.export(tmp_in, format="wav")
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
    sc= f"""
    <script>
    function copyText(txt){{
        navigator.clipboard.writeText(txt).then(
            ()=>alert('Texte copié!'),
            (err)=>alert('Échec copie: '+err)
        );
    }}
    </script>
    <button onclick="copyText(`{text}`)" style="margin-top:5px;">Copier</button>
    """
    st.components.v1.html(sc)

def display_history_side():
    st.sidebar.header("Historique")
    st.sidebar.write("---")
    tlist= load_transcriptions()
    if not tlist:
        st.sidebar.info("Aucune transcription enregistrée.")
        return
    st.sidebar.write(f"Total: {len(tlist)} transcriptions.")
    for item in tlist[::-1][:8]:
        st.sidebar.markdown(f"- {item.get('audio_name','?')} / {item.get('date','?')}")

def main():
    init_transcriptions()
    display_history_side()

    st.title("NBL Audio : Transcription")
    st.markdown("Choisissez votre source audio (fichier, micro, ou multi), configurez les options (barre latérale), puis lancez la transcription.")

    # Barre latérale : transformations, double mode, modèle, etc.
    st.sidebar.header("Options")
    remove_sil= st.sidebar.checkbox("Supprimer les silences",False)
    speed_factor= st.sidebar.slider("Accélération Audio",0.5,4.0,1.0,0.1)
    double_mode= st.sidebar.checkbox("Transcription Double (Nova2 + Whisper)",False)

    st.sidebar.write("---")
    st.sidebar.header("Modèle (si simple)")
    model_choice= st.sidebar.radio("Modèle IA:", ["Nova 2","Whisper Large"])
    model_map= {"Nova 2":"nova-2", "Whisper Large":"whisper-large"}

    st.sidebar.write("---")
    st.sidebar.header("Langue")
    lang_choice= st.sidebar.radio("Langue:", ["fr","en"])

    # Choix du mode
    st.write("## Mode d'Entrée")
    mode_in= st.radio("", ["Fichier (Upload)","Micro (Enregistrement)","Multi-Fichiers","Multi-Micro"])

    segments=[]  # liste de (AudioSegment, nom)

    # Collecte
    if mode_in=="Fichier (Upload)":
        upf= st.file_uploader("Fichier audio", type=["mp3","wav","m4a","ogg","webm"])
        if upf:
            st.audio(upf, format=upf.type)
            rename= st.text_input("Nom (optionnel)", upf.name)
            segments.append((upf, rename))

    elif mode_in=="Micro (Enregistrement)":
        mic= st.audio_input("Enregistrer micro")
        if mic:
            st.audio(mic, format=mic.type)
            rename= st.text_input("Nom (optionnel)","micro.wav")
            segments.append((mic, rename))

    elif mode_in=="Multi-Fichiers":
        st.write("Chargez plusieurs fichiers d'un coup :")
        many_files= st.file_uploader("Import multiple", accept_multiple_files=True, type=["mp3","wav","m4a","ogg","webm"])
        if many_files:
            for i,fobj in enumerate(many_files):
                st.audio(fobj, format=fobj.type)
                rename= st.text_input(f"Nom Fichier #{i+1}", fobj.name, key=f"multif_{i}")
                segments.append((fobj, rename))

    else:
        # Multi micro
        st.write("Combien de micros souhaitez-vous enregistrer ?")
        n_mic= st.number_input("Nombre de micros", min_value=1, max_value=10, value=1, step=1)
        for i in range(n_mic):
            micX= st.audio_input(f"Micro #{i+1}", key=f"mic_{i}")
            if micX:
                st.audio(micX, format=micX.type)
                rename= st.text_input(f"Nom micro #{i+1}", f"micro_{i}.wav", key=f"namem_{i}")
                segments.append((micX, rename))

    # Bouton "Transcrire"
    if len(segments)>0 and st.button("Transcrire"):
        # On applique transformations => on transcrit
        trans_list= load_transcriptions()
        startT= time.time()
        for idx,(srcObj, rename) in enumerate(segments):
            # Sauver local
            local_path= f"temp_input_{idx}.wav"
            with open(local_path,"wb") as ff:
                ff.write(srcObj.read())
            # Charger AudioSegment
            seg= AudioSegment.from_file(local_path)
            if remove_sil:
                seg= remove_silences_classic(seg)
            if abs(speed_factor-1.0)>1e-2:
                seg= accelerate_ffmpeg(seg, speed_factor)
            # Export final
            seg_path= f"temp_final_{idx}.wav"
            seg.export(seg_path, format="wav")
            # Double ou simple
            if double_mode:
                st.info(f"[Segment #{idx+1}] IA Nova2 en cours...")
                text1= nova_api.transcribe_audio(seg_path, st.secrets["NOVA1"], lang_choice,"nova-2")
                st.success("Première transcription ok (Nova2).")
                st.info(f"[Segment #{idx+1}] IA WhisperLarge en cours...")
                text2= nova_api.transcribe_audio(seg_path, st.secrets["NOVA2"], lang_choice,"whisper-large")
                st.success("Deuxième transcription ok (Whisper).")
                colL,colR= st.columns(2)
                with colL:
                    st.subheader(f"Nova2 - Segment #{idx+1}")
                    st.text_area("", text1, height=150)
                    copy_to_clipboard(text1)
                with colR:
                    st.subheader(f"WhisperLarge - Segment #{idx+1}")
                    st.text_area("", text2, height=150)
                    copy_to_clipboard(text2)
                trans_list.append({
                    "audio_name": rename,
                    "double_mode": True,
                    "transcript_nova2": text1,
                    "transcript_whisper": text2,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                # Simple
                if model_choice=="Nova 2":
                    keyUsed= st.secrets["NOVA1"]
                    modelUsed= "nova-2"
                else:
                    keyUsed= st.secrets["NOVA2"]
                    modelUsed= "whisper-large"
                st.info(f"[Segment #{idx+1}] Transcription en cours ({model_choice})...")
                txt= nova_api.transcribe_audio(seg_path, keyUsed, lang_choice, modelUsed)
                st.success("Transcription terminée.")
                st.text_area("", txt, height=150)
                copy_to_clipboard(txt)
                trans_list.append({
                    "audio_name": rename,
                    "double_mode": False,
                    "model": modelUsed,
                    "transcript": txt,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            # Nettoyage
            try: os.remove(local_path)
            except: pass
            try: os.remove(seg_path)
            except: pass
        save_transcriptions(trans_list)
        elapsed= time.time()-startT
        st.success(f"Tous les segments ont été transcrits en {elapsed:.1f}s.")

if __name__=="__main__":
    main()
