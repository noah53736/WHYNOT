import streamlit as st
import os
import io
import time
import random
import string
import json
import subprocess
from datetime import datetime
from pydub import AudioSegment, silence
import nova_api

# -- IMPORTANT: set_page_config au tout début pour éviter l'erreur --
st.set_page_config(page_title="NBL Audio", layout="wide")

HISTORY_FILE = "historique.json"
CREDITS_FILE = "credits.json"

MIN_SILENCE_LEN = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
COST_PER_SEC = 0.007

# =====================================================================
# Fonctions de gestion d'historique et crédits
# =====================================================================
def init_history():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f, indent=4)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        init_history()
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(hist):
    with open(HISTORY_FILE, "w") as f:
        json.dump(hist, f, indent=4)

def load_credits():
    if os.path.exists(CREDITS_FILE):
        with open(CREDITS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_credits(c):
    with open(CREDITS_FILE, "w") as f:
        json.dump(c, f, indent=4)

# =====================================================================
# Utils divers
# =====================================================================
def generate_alias(length=5):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

def human_time(sec: float) -> str:
    s = int(sec)
    if s < 60: return f"{s}s"
    elif s < 3600:
        m, s2 = divmod(s, 60)
        return f"{m}m{s2}s"
    else:
        h, r = divmod(s, 3600)
        m, s3 = divmod(r, 60)
        return f"{h}h{m}m{s3}s"

def accelerate_ffmpeg(audio_seg: AudioSegment, factor: float) -> AudioSegment:
    if abs(factor - 1.0) < 1e-2:
        return audio_seg
    tmp_in = "temp_in_acc.wav"
    tmp_out = "temp_out_acc.wav"
    audio_seg.export(tmp_in, format="wav")
    remain = factor
    filters = []
    while remain > 2.0:
        filters.append("atempo=2.0")
        remain /= 2.0
    while remain < 0.5:
        filters.append("atempo=0.5")
        remain /= 0.5
    filters.append(f"atempo={remain}")
    f_str = ",".join(filters)
    cmd = ["ffmpeg", "-y", "-i", tmp_in, "-filter:a", f_str, tmp_out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new_seg = AudioSegment.from_file(tmp_out, format="wav")
    try:
        os.remove(tmp_in)
        os.remove(tmp_out)
    except:
        pass
    return new_seg

def remove_silences_classic(audio_seg: AudioSegment,
                            min_silence_len=MIN_SILENCE_LEN,
                            silence_thresh=SIL_THRESH_DB,
                            keep_silence=KEEP_SIL_MS):
    segs = silence.split_on_silence(
        audio_seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    if not segs:
        return audio_seg
    combined = segs[0]
    for s in segs[1:]:
        combined = combined.append(s, crossfade=0)
    return combined

def copy_to_clipboard(text):
    sc = f"""
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

def select_api_keys(credits, key_ids, duration_sec, cost_per_sec=COST_PER_SEC):
    needed = duration_sec * cost_per_sec
    sel = []
    for k in key_ids:
        if credits.get(k,0)>= needed:
            sel.append((k, needed))
            if len(sel)==2: break
    return sel

def display_history():
    st.sidebar.header("Historique")
    st.sidebar.write("---")
    h = st.session_state.get("history", [])
    if h:
        tb = []
        for e in h:
            tb.append({
                "Alias/Nom": e["Alias/Nom"],
                "Méthode": e["Méthode"],
                "Modèle": e["Modèle"],
                "Durée": e["Durée"],
                "Temps": e["Temps"],
                "Coût": e["Coût"],
                "Date": e["Date"]
            })
        st.sidebar.table(tb[::-1])
        st.sidebar.write("### Aperçus Audio")
        for e2 in reversed(h[-3:]):
            st.sidebar.markdown(f"**{e2['Alias/Nom']}** – {e2['Date']}")
            ab = bytes.fromhex(e2["Audio Binaire"])
            st.sidebar.audio(ab, format="audio/wav")
    else:
        st.sidebar.info("Aucun historique pour le moment.")

# =====================================================================
# main
# =====================================================================
def main():
    st.title("NBL Audio")
    st.markdown("Un logiciel conçu par M. NB pour une **transcription** pratique !")

    if "history" not in st.session_state:
        st.session_state["history"] = load_history()
    hist = st.session_state["history"]
    creds = load_credits()

    # -- Barre latérale: Accessibilité, transformations, etc. --
    st.sidebar.header("Accessibilité et Transformations")
    double_trans = st.sidebar.checkbox("Transcrire simultanément avec deux IA pour un résultat plus précis", False)
    remove_sil = st.sidebar.checkbox("Supprimer Silences")
    speed_factor = st.sidebar.slider("Accélération Audio",0.5,4.0,1.0,0.1)
    
    # Choix du modèle si on veut simple transcription
    st.sidebar.write("---")
    st.sidebar.header("Choix du Modèle (monoposte)")
    model_choice = st.sidebar.radio("Modèle IA :", ["Nova 2","Whisper Large"])
    model_map = {"Nova 2":"nova-2","Whisper Large":"whisper-large"}
    chosen_model = model_map.get(model_choice,"nova-2")
    
    st.sidebar.write("---")
    st.sidebar.header("Langue")
    lang_choice = st.sidebar.radio("Sélection Langue :", ["fr","en"])

    st.sidebar.write("---")
    if st.sidebar.button("Vider l'Historique"):
        st.session_state["history"]=[]
        save_history([])
        st.sidebar.success("Historique effacé.")
        st.experimental_rerun()

    # Affichage Historique (barre latérale)
    display_history()

    st.write("## Sélection du Mode d'Entrée")
    # Multi-fichiers/multi-audio
    input_mode = st.radio("Mode :", ["1 Fichier/Micro","Multi-Fichiers/Multi-Audios"])
    
    audio_data_list = []
    file_names = []

    if input_mode=="1 Fichier/Micro":
        method = st.radio("Type d'Entrée :", ["Fichier (Upload)","Micro (Enregistrement)"])
        if method=="Fichier (Upload)":
            upf = st.file_uploader("Importez un audio (mp3/wav/m4a/ogg/webm)", type=["mp3","wav","m4a","ogg","webm"])
            if upf:
                if upf.size>200*1024*1024:
                    st.warning("Fichier trop volumineux (>200MB).")
                else:
                    audio_data_list=[upf.read()]
                    file_names=[st.text_input("Nom du Fichier (Optionnel)")]
                    st.audio(upf, format=upf.type)
        else:
            mic_in = st.audio_input("Enregistrement Micro")
            if mic_in:
                audio_data_list=[mic_in.read()]
                file_names=[st.text_input("Nom du Fichier (Optionnel)")]
                st.audio(mic_in, format=mic_in.type)

    else:
        # MULTI-FICHIERS
        st.write("Uploadez jusqu'à 5 fichiers ou enregistrez 5 pistes micro.")
        for i in range(5):
            col_left, col_right= st.columns([3,1])
            with col_left:
                upf_multi = st.file_uploader(f"Audio #{i+1}", type=["mp3","wav","m4a","ogg","webm"], key=f"multi_{i}")
                if upf_multi:
                    if upf_multi.size>200*1024*1024:
                        st.warning(f"Audio #{i+1}: trop volumineux.")
                    else:
                        audio_data_list.append(upf_multi.read())
                        fname = st.text_input(f"Nom Fichier #{i+1} (Optionnel)", key=f"name_{i}")
                        file_names.append(fname if fname else "")
                        st.audio(upf_multi, format=upf_multi.type)
            with col_right:
                micX= st.audio_input(f"Micro #{i+1}", key=f"mic_{i}")
                if micX:
                    audio_data_list.append(micX.read())
                    fname= st.text_input(f"Nom Micro #{i+1} (Optionnel)", key=f"name_m_{i}")
                    file_names.append(fname if fname else "")
                    st.audio(micX, format=micX.type)
        # Supprimer entrées vides
        adl=[]
        fnm=[]
        for i, ad in enumerate(audio_data_list):
            if ad and len(ad)>0:
                adl.append(ad)
                fnm.append(file_names[i] if i<len(file_names) else "")
        audio_data_list= adl
        file_names= fnm

    # Bouton Effacer
    if st.button("Effacer l'Entrée"):
        st.experimental_rerun()

    if len(audio_data_list)==0:
        st.warning("Aucune entrée audio sélectionnée.")
        return

    # Appliquer transformations => final_aud_list
    final_segments=[]
    original_durations=[]
    for idx, adata in enumerate(audio_data_list):
        try:
            seg= AudioSegment.from_file(io.BytesIO(adata))
            orig_sec= len(seg)/1000.0
            original_durations.append(orig_sec)
            if remove_sil:
                seg= remove_silences_classic(seg)
            if abs(speed_factor-1.0)>1e-2:
                seg= accelerate_ffmpeg(seg, speed_factor)
            final_segments.append(seg)
        except Exception as e:
            st.error(f"[Segment #{idx+1}] Erreur prétraitement : {e}")
            return

    st.success(f"{len(final_segments)} segment(s) prêt(s) pour la transcription.")
    for idx, seg in enumerate(final_segments):
        st.write(f"Segment #{idx+1}: Original={human_time(original_durations[idx])} / Final={human_time(len(seg)/1000)}")

    # Aperçu audio transformé (on en met un seul? ou plusieurs?)
    # On peut tous les afficher
    for idx, seg in enumerate(final_segments):
        buf= io.BytesIO()
        seg.export(buf, format="wav")
        st.write(f"### Aperçu Segment #{idx+1}")
        st.audio(buf.getvalue(), format="audio/wav")

    # Lancer la transcription
    if st.button("Transcrire Maintenant"):
        creds= load_credits()
        st.info("Lancement de la transcription...")
        startAll= time.time()
        
        # 1) Calcul du temps total => sum(len(segments))
        total_dur= sum([len(s)/1000 for s in final_segments])
        # 2) Sélection des clés
        # Si double => 2 clés + Double transcription
        # Sinon => 1 clé + single transcription
        if double_trans:
            sk= select_api_keys(creds, list(st.secrets.keys()), total_dur)
            if len(sk)<2:
                st.error("Pas assez de clés API ou crédits pour double transcription.")
                return
            k1_id, cost1= sk[0]
            k2_id, cost2= sk[1]
            api1= st.secrets[k1_id]
            api2= st.secrets[k2_id]
            res1=[]
            res2=[]
            # On transcrit chaque segment
            for i, seg in enumerate(final_segments):
                st.write(f"Transcription double du segment #{i+1}...")
                temp_path= f"temp_multi_{i}.wav"
                seg.export(temp_path, format="wav")
                # Lancer la transcription 1 (Nova2)
                partial1= nova_api.transcribe_audio(temp_path, api1, language=lang_choice, model_name="nova-2")
                st.write(f"Segment #{i+1} (Nova2) terminé, en attente du whisper-large")
                partial2= nova_api.transcribe_audio(temp_path, api2, language=lang_choice, model_name="whisper-large")
                res1.append(partial1)
                res2.append(partial2)
                try: os.remove(temp_path)
                except: pass
            # On a res1 => tous segments, res2 => tous segments
            tNova= "\n".join(res1)
            tWhisp= "\n".join(res2)
            elapsed= time.time()-startAll
            st.success(f"Double transcription multi terminée en {elapsed:.1f}s.")
            colL, colR= st.columns(2)
            with colL:
                st.subheader("Nova 2 (global)")
                st.text_area("Texte complet Nova2", tNova, height=180)
                copy_to_clipboard(tNova)
            with colR:
                st.subheader("Whisper Large (global)")
                st.text_area("Texte complet Whisper", tWhisp, height=180)
                copy_to_clipboard(tWhisp)
            st.write(f"Temps: {human_time(elapsed)} | Coût estimé= ~${cost1+cost2:.2f}")
            # Ajout historique
            a1= generate_alias(6)
            a2= generate_alias(6)
            e1= {
                "Alias/Nom": a1,
                "Méthode": "Nova 2 (multi-chunks)",
                "Modèle": "nova-2",
                "Durée": human_time(total_dur),
                "Temps": human_time(elapsed),
                "Coût": f"${cost1:.2f}",
                "Transcription": tNova,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Audio Binaire": ""  # multi files => on stocke rien ou un seul?
            }
            e2= {
                "Alias/Nom": a2,
                "Méthode": "Whisper Large (multi-chunks)",
                "Modèle": "whisper-large",
                "Durée": human_time(total_dur),
                "Temps": human_time(elapsed),
                "Coût": f"${cost2:.2f}",
                "Transcription": tWhisp,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Audio Binaire": ""
            }
            st.session_state["history"].extend([e1,e2])
            save_history(st.session_state["history"])
            st.success("Historique mis à jour.")
            creds[k1_id]-= cost1
            creds[k2_id]-= cost2
            save_credits(creds)
        else:
            # Simple transcription => 1 clé
            selS= select_api_keys(creds, list(st.secrets.keys()), total_dur)
            if len(selS)<1:
                st.error("Pas assez de crédits/clés API.")
                return
            kid, cst= selS[0]
            apk= st.secrets[kid]
            partials=[]
            for i, seg in enumerate(final_segments):
                st.write(f"Transcription segment #{i+1} (single mode, {model_map.get(model_choice)})...")
                tP= f"temp_multi_{i}.wav"
                seg.export(tP, format="wav")
                pr= nova_api.transcribe_audio(tP, apk, lang_choice, model_map.get(model_choice,"nova-2"))
                partials.append(pr)
                try: os.remove(tP)
                except: pass
            full_txt= "\n".join(partials)
            elapsed= time.time()-startAll
            st.success(f"Multi segments terminés en {elapsed:.1f}s.")
            st.text_area("Texte complet", full_txt, height=180)
            copy_to_clipboard(full_txt)
            st.write(f"Temps: {human_time(elapsed)} / Coût= ~${cst:.2f}")
            aZ= generate_alias(6)
            eZ= {
                "Alias/Nom": aZ,
                "Méthode": model_map.get(model_choice,"nova-2"),
                "Modèle": model_map.get(model_choice,"nova-2"),
                "Durée": human_time(total_dur),
                "Temps": human_time(elapsed),
                "Coût": f"${cst:.2f}",
                "Transcription": full_txt,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Audio Binaire": ""
            }
            st.session_state["history"].append(eZ)
            save_history(st.session_state["history"])
            st.success("Historique mis à jour.")
            creds[kid]-= cst
            save_credits(creds)

def main_wrapper():
    main()

if __name__=="__main__":
    main_wrapper()
