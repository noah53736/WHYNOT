import streamlit as st
import pandas as pd
import os
import io
import time
import random
import string
import subprocess
from datetime import datetime
import requests
import traceback
import shutil

# Pour la partie audio
from pydub import AudioSegment, silence

# PyTorch + Whisper
import torch
import whisper

###############################################################################
# CONFIGURATION FFMPEG / FFPROBE
###############################################################################
# Force pydub à utiliser le ffmpeg/ffprobe installé via packages.txt
AudioSegment.converter = shutil.which("ffmpeg")
AudioSegment.ffprobe   = shutil.which("ffprobe")

###############################################################################
# FONCTION DE TRANSCRIPTION NOVA
###############################################################################
def transcribe_nova_one_shot(
    file_path: str,
    dg_api_key: str,      # la clé Nova (ex: st.secrets["NOVA"] ou os.getenv("NOVA")
    language: str = "fr",
    model_name: str = "nova-2",
    punctuate: bool = True,
    numerals: bool = True
):
    """
    Envoie le fichier complet à Deepgram (Nova ou Whisper-Cloud) et retourne la transcription (str).
    """
    temp_in = "temp_nova_in.wav"

    # Conversion du fichier audio en WAV 16kHz mono
    audio_seg = AudioSegment.from_file(file_path)
    audio_16k = audio_seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio_16k.export(temp_in, format="wav")

    # Prépare la requête
    # Exemple: ?language=fr&model=whisper-large&punctuate=true&numerals=true
    params = []
    params.append(f"language={language}")
    params.append(f"model={model_name}")
    if punctuate:
        params.append("punctuate=true")
    if numerals:
        params.append("numerals=true")

    qs = "?" + "&".join(params)
    url = "https://api.deepgram.com/v1/listen" + qs

    try:
        with open(temp_in, "rb") as f:
            buf = f.read()
        heads = {
            "Authorization": f"Token {dg_api_key}",
            "Content-Type": "audio/wav"
        }
        resp = requests.post(url, headers=heads, data=buf)
        if resp.status_code == 200:
            j = resp.json()
            alt = j.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0]
            txt = alt.get("transcript", "")
            return txt
        else:
            st.error(f"[Nova] Erreur {resp.status_code} : {resp.text}")
            return ""
    except Exception as e:
        st.error(f"[Nova] Exception : {e}")
        traceback.print_exc()
        return ""
    finally:
        try:
            os.remove(temp_in)
        except:
            pass

###############################################################################
# CONSTANTES + GESTION HISTORIQUE
###############################################################################
HISTORY_FILE= "historique.csv"

def init_history():
    if not os.path.exists(HISTORY_FILE):
        df_init = pd.DataFrame(columns=[
            "Alias","Méthode","Modèle","Durée","Temps","Coût",
            "Transcription","Date","Audio Binaire"
        ])
        df_init.to_csv(HISTORY_FILE, index=False)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        init_history()
    return pd.read_csv(HISTORY_FILE)

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)

def generate_alias(length=5):
    return "".join(random.choices(string.ascii_uppercase+string.digits, k=length))

def human_time(sec: float)-> str:
    sec=int(sec)
    if sec<60:
        return f"{sec}s"
    elif sec<3600:
        m,s= divmod(sec,60)
        return f"{m}m{s}s"
    else:
        h,r= divmod(sec,3600)
        m,s= divmod(r,60)
        return f"{h}h{m}m{s}s"

def remove_euh_in_text(text: str):
    words= text.split()
    cleaned=[]
    cnt=0
    for w in words:
        wlow= w.lower().strip(".,!?;:")
        if "euh" in wlow or "heu" in wlow:
            cnt+=1
        else:
            cleaned.append(w)
    return (" ".join(cleaned), cnt)

###############################################################################
# ACCELERATION (TIME-STRETCH)
###############################################################################
def accelerate_ffmpeg(audio_seg: AudioSegment, factor: float)-> AudioSegment:
    """
    Accélère ou ralentit le signal via ffmpeg (atempo).
    factor=2 => x2 plus rapide
    factor=0.5 => 2x plus lent
    """
    if abs(factor - 1.0) < 1e-2:
        return audio_seg  # Pas de changement

    tmp_in="temp_acc_in.wav"
    tmp_out="temp_acc_out.wav"
    audio_seg.export(tmp_in,format="wav")

    remain= factor
    filters=[]
    # Pour atempo, la valeur doit être entre 0.5 et 2.0
    while remain>2.0:
        filters.append("atempo=2.0")
        remain/=2.0
    while remain<0.5:
        filters.append("atempo=0.5")
        remain/=0.5
    filters.append(f"atempo={remain}")
    f_str=",".join(filters)

    cmd=[
        "ffmpeg","-y","-i", tmp_in,
        "-filter:a", f_str,
        tmp_out
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new_seg= AudioSegment.from_file(tmp_out, format="wav")
    try:
        os.remove(tmp_in)
        os.remove(tmp_out)
    except:
        pass
    return new_seg

###############################################################################
# SUPPRESSION DE SILENCES (DOUCE)
###############################################################################
MIN_SIL_MS=700
SIL_THRESH_DB=-35
KEEP_SIL_MS=50
CROSSFADE_MS=50

def remove_silences_smooth(audio_seg: AudioSegment,
                           min_sil_len=MIN_SIL_MS,
                           silence_thresh=SIL_THRESH_DB,
                           keep_sil_ms=KEEP_SIL_MS,
                           crossfade_ms=CROSSFADE_MS):
    """
    Découpe la piste sur les silences, puis recolle le tout
    en fondant les transitions (crossfade).
    """
    segs = silence.split_on_silence(
        audio_seg,
        min_sil_len=min_sil_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_sil_ms
    )
    if not segs:
        return audio_seg  # Rien trouvé, ou piste vide

    combined = segs[0]
    for s in segs[1:]:
        combined = combined.append(s, crossfade=crossfade_ms)
    return combined

###############################################################################
# LOCAL WHISPER
###############################################################################
@st.cache_resource  # Si tu as Streamlit >= 1.18, sinon remplace par @st.cache
def load_local_model(model_name:str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"[Local Whisper] Chargement du modèle {model_name} sur {device}...")
    return whisper.load_model(model_name, device=device)

def transcribe_local(file_path:str, model_name:str)-> str:
    model = load_local_model(model_name)
    result = model.transcribe(file_path)
    return result["text"]

###############################################################################
# OPENAI
###############################################################################
WHISPER_API_COST_PER_MINUTE= 0.006
def transcribe_openai(file_path:str, openai_key:str)-> str:
    """
    Appel API OpenAI Whisper endpoint /v1/audio/transcriptions
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    heads = {"Authorization": f"Bearer {openai_key}"}
    with open(file_path,"rb") as f:
        files={"file": f}
        data={"model":"whisper-1","response_format":"text"}
        r= requests.post(url, headers=heads, files=files, data=data)
    if r.status_code==200:
        return r.text
    else:
        st.error(f"[OpenAI] Erreur {r.status_code} : {r.text}")
        return ""

###############################################################################
# DEEPGRAM / NOVA
###############################################################################
NOVA_COST_PER_MINUTE= 0.007

###############################################################################
# MAIN STREAMLIT
###############################################################################
def main():
    st.set_page_config(page_title="NBL Audio – 3 modes", layout="wide")
    st.title("NBL Audio – Local / OpenAI / Nova")

    # Initialiser / Charger l'historique
    if "hist_df" not in st.session_state:
        df = load_history()
        st.session_state["hist_df"] = df
    hist_df = st.session_state["hist_df"]

    # 1) Choix du mode
    st.sidebar.header("Mode de Transcription")
    mode = st.sidebar.radio(
        "Choix :",
        ["Local Whisper","OpenAI Whisper","Nova (Deepgram)"]
    )

    # 2) Lecture des clés (en variables d'env, secrets, etc.)
    # OpenAI key
    openai_key = os.getenv("APIKEY","")  # "APIKEY" en majuscules

    # Nova keys : "NOVA", "NOVA2", "NOVA3" => on propose un select
    nova_key_choice = st.sidebar.selectbox(
        "Clé Nova à utiliser",
        ["NOVA","NOVA2","NOVA3"],
        index=0
    )
    chosen_nova_key = os.getenv(nova_key_choice,"")

    # 3) Paramètres pour chaque mode
    local_model = None
    if mode=="Local Whisper":
        local_model= st.sidebar.selectbox(
            "Modèle local Whisper",
            ["tiny","base","small","medium","large","large-v2"],
            index=1
        )

    lang_nova="fr"
    model_nova="nova-2"
    punct_nova=True
    numerals_nova=True
    if mode=="Nova (Deepgram)":
        st.sidebar.subheader("Deepgram Nova – Paramètres")
        lang_nova= st.sidebar.text_input("Langue (ex: 'fr', 'en-US')", "fr")
        model_nova= st.sidebar.selectbox(
            "Modèle Deepgram",
            ["nova-2","whisper-base","whisper-medium","whisper-large"],
            index=0
        )
        punct_nova= st.sidebar.checkbox("Ponctuation (punctuate)", True)
        numerals_nova= st.sidebar.checkbox("Nombres => chiffres (numerals)", True)

    # 4) Transformations audio
    st.sidebar.write("---")
    st.sidebar.header("Transformations")
    remove_sil = st.sidebar.checkbox("Supprimer silences (douce)", False)
    speed_factor= st.sidebar.slider("Accélération (time-stretch)", 0.5, 4.0, 1.0, 0.1)
    remove_euh_box= st.sidebar.checkbox("Retirer 'euh' ?", False)

    # Historique sur la sidebar
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    disp_cols= ["Alias","Méthode","Modèle","Durée","Temps","Coût","Transcription","Date"]
    if not hist_df.empty:
        show_cols= [c for c in disp_cols if c in hist_df.columns]
        st.sidebar.dataframe(hist_df[show_cols][::-1], use_container_width=True)
    else:
        st.sidebar.info("Historique vide.")

    st.sidebar.write("### Derniers Audios (3)")
    if not hist_df.empty:
        last_auds= hist_df[::-1].head(3)
        for idx,row in last_auds.iterrows():
            ab= row.get("Audio Binaire",None)
            if isinstance(ab, bytes):
                st.sidebar.write(f"**{row.get('Alias','?')}** – {row.get('Date','?')}")
                st.sidebar.audio(ab, format="audio/wav")

    ###########################################################################
    # SECTION PRINCIPALE
    ###########################################################################
    st.write("## Source Audio")
    audio_data = None
    input_choice = st.radio("Fichier ou Micro ?", ["Fichier","Micro"])
    if input_choice=="Fichier":
        upf= st.file_uploader(
            "Fichier audio (mp3,wav,m4a,ogg,webm)",
            type=["mp3","wav","m4a","ogg","webm"]
        )
        if upf:
            audio_data= upf.read()
    else:
        mic_in= st.audio_input("Enregistrement micro")
        if mic_in:
            audio_data= mic_in.read()

    if audio_data:
        # On enregistre un wav intermédiaire
        try:
            with open("temp_original.wav","wb") as foo:
                foo.write(audio_data)

            aud= AudioSegment.from_file("temp_original.wav")
            original_sec= len(aud)/1000.0
            st.write(f"Durée d'origine : {human_time(original_sec)}")

            final_aud= aud
            # 1) suppression silences
            if remove_sil:
                final_aud= remove_silences_smooth(final_aud)
            # 2) accélération
            if abs(speed_factor-1.0)>1e-2:
                final_aud= accelerate_ffmpeg(final_aud, speed_factor)

            # On stocke le son transformé en RAM
            bufp= io.BytesIO()
            final_aud.export(bufp, format="wav")
            st.write("### Aperçu Audio transformé")
            st.audio(bufp.getvalue(), format="audio/wav")

        except Exception as e:
            st.error(f"Erreur chargement/preproc : {e}")

    # BOUTON DE TRANSCRIPTION
    if audio_data:
        if st.button("Transcrire maintenant"):
            start_t= time.time()

            # Enregistrer l'audio transformé
            with open("temp_input.wav","wb") as f:
                final_aud.export(f, format="wav")

            final_txt = ""
            cost_str = "0.00 €"
            final_sec= len(final_aud)/1000.0

            # --- Mode 1 : Local Whisper
            if mode=="Local Whisper":
                if not local_model:
                    st.error("Aucun modèle local sélectionné.")
                    return
                txt= transcribe_local("temp_input.wav", local_model)
                final_txt= txt
                cost_str= "0.00 €"
                st.info("[Local Whisper] Coût=0€")

            # --- Mode 2 : OpenAI Whisper
            elif mode=="OpenAI Whisper":
                if not openai_key:
                    st.error("Clé 'APIKEY' (OpenAI) manquante en variable d'env ou secrets.")
                    return
                txt= transcribe_openai("temp_input.wav", openai_key)
                final_txt= txt
                mn= final_sec/60.0
                cost_val= round(mn * WHISPER_API_COST_PER_MINUTE,3)
                cost_str= f"{cost_val} $"
                st.info(f"[OpenAI] ~{cost_str}")

            # --- Mode 3 : Nova (Deepgram)
            else:
                if not chosen_nova_key:
                    st.error(f"La clé '{nova_key_choice}' est vide ?")
                    return
                txt= transcribe_nova_one_shot(
                    file_path="temp_input.wav",
                    dg_api_key= chosen_nova_key,
                    language= lang_nova,
                    model_name= model_nova,
                    punctuate= punct_nova,
                    numerals= numerals_nova
                )
                final_txt= txt
                mn= final_sec/60.0
                cost_val= round(mn * NOVA_COST_PER_MINUTE,3)
                cost_str= f"{cost_val} $"
                st.info(f"[Nova] ~{cost_str}")

            # Option : retirer "euh"
            if remove_euh_box:
                new_txt, ccc= remove_euh_in_text(final_txt)
                final_txt= new_txt
                st.write(f"({ccc} occurrences de 'euh' retirées)")

            total_proc= time.time()-start_t
            total_str= human_time(total_proc)

            # Gains : si on a réduit la durée par le time-stretch, etc.
            gain_sec=0
            if final_sec< original_sec:
                gain_sec= original_sec- final_sec
            gain_str= human_time(gain_sec)

            st.success(f"Durée finale : {human_time(final_sec)} (gagné {gain_str}) | "
                       f"Temps effectif: {total_str} | Coût={cost_str}")

            st.write("## Résultat Final")
            st.text_area("Texte :", final_txt, height=200)

            st.download_button(
                "Télécharger",
                data= final_txt.encode("utf-8"),
                file_name="transcription.txt"
            )

            # Mettre à jour l'historique
            alias= generate_alias(6)
            audio_buf= io.BytesIO(audio_data)

            used_model= ""
            if mode=="Local Whisper":
                used_model= local_model
            elif mode=="OpenAI Whisper":
                used_model= "whisper-1"
            else:
                used_model= model_nova

            new_row={
                "Alias": alias,
                "Méthode": mode,
                "Modèle": used_model,
                "Durée": f"{human_time(original_sec)}",
                "Temps": total_str,
                "Coût": cost_str,
                "Transcription": final_txt,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Audio Binaire": audio_buf.getvalue()
            }
            hist_df= pd.concat([hist_df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state["hist_df"]= hist_df
            save_history(hist_df)

            st.info(f"Historique mis à jour. Alias={alias}")

def main_wrapper():
    main()

if __name__=="__main__":
    main_wrapper()
