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

HISTORY_FILE = "historique.json"
CREDITS_FILE = "credits.json"

MIN_SILENCE_LEN = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
COST_PER_SEC = 0.007

# -- Place set_page_config at the very top! --
st.set_page_config(page_title="N-B-L Audio", layout="wide")

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
    cmd = ["ffmpeg","-y","-i", tmp_in,"-filter:a", f_str, tmp_out]
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

def select_api_keys(credits, key_ids, duration_sec, cost_per_sec=COST_PER_SEC, used_keys=[]):
    cost = duration_sec * cost_per_sec
    selected = []
    for key_id in key_ids:
        if key_id in used_keys:
            continue
        if credits.get(key_id, 0) >= cost:
            selected.append((key_id, cost))
            if len(selected) == 2:
                break
    return selected

def copy_to_clipboard(text):
    script_html = f"""
    <script>
        function copyText(txt) {{
            navigator.clipboard.writeText(txt).then(function() {{
                alert('Transcription copiée dans le presse-papiers!');
            }}, function(err) {{
                alert('Échec de la copie : ' + err);
            }});
        }}
    </script>
    <button onclick="copyText(`{text}`)" style="margin:5px;">Copier</button>
    """
    st.components.v1.html(script_html)

def load_credits():
    if os.path.exists(CREDITS_FILE):
        with open(CREDITS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_credits(c):
    with open(CREDITS_FILE, "w") as f:
        json.dump(c, f, indent=4)

def display_history():
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    h = st.session_state.get("history", [])
    if h:
        table = []
        for en in h:
            table.append({
                "Alias/Nom": en["Alias/Nom"],
                "Méthode": en["Méthode"],
                "Modèle": en["Modèle"],
                "Durée": en["Durée"],
                "Temps": en["Temps"],
                "Coût": en["Coût"],
                "Date": en["Date"]
            })
        st.sidebar.table(table[::-1])
        st.sidebar.write("### Aperçus Audio")
        for en in reversed(h[-3:]):
            st.sidebar.markdown(f"**{en['Alias/Nom']}** – {en['Date']}")
            ab = bytes.fromhex(en["Audio Binaire"])
            st.sidebar.audio(ab, format="audio/wav")
    else:
        st.sidebar.info("Historique vide.")

def main():
    st.title("N-B-L Audio : Transcription Grand Public")
    if "history" not in st.session_state:
        st.session_state["history"] = load_history()
    history = st.session_state["history"]
    credits = load_credits()

    st.write("## Options de Transcription")
    col1, col2 = st.columns([1,1])
    with col1:
        model_selection = st.selectbox("Modèle :", ["Nova 2", "Whisper Large"])
    with col2:
        language_selection = st.selectbox("Langue :", ["fr", "en"])

    model_map = {"Nova 2":"nova-2","Whisper Large":"whisper-large"}
    chosen_model = model_map.get(model_selection,"nova-2")

    accessibility = st.checkbox("Double Transcription (Accessibilité)")

    st.write("---")
    st.write("### Transformations Audio")
    remove_sil = st.checkbox("Supprimer Silences")
    speed_factor = st.slider("Accélération", 0.5,4.0,1.0,0.1)

    st.write("## Entrée Audio")
    audio_data = None
    file_name = None

    input_type = st.radio("Type d'Entrée :", ["Fichier (Upload)", "Micro (Enregistrement)"])
    if input_type=="Fichier (Upload)":
        upf = st.file_uploader("Importer l'audio", type=["mp3","wav","m4a","ogg","webm"])
        if upf:
            if upf.size>200*1024*1024:
                st.warning("Fichier > 200MB (limite Streamlit).")
            else:
                audio_data = upf.read()
                st.audio(upf, format=upf.type)
                file_name = st.text_input("Nom du Fichier (Optionnel)")
    else:
        mic_input = st.audio_input("Micro")
        if mic_input:
            audio_data = mic_input.read()
            st.audio(mic_input, format=mic_input.type)

    cA,cB = st.columns(2)
    with cA:
        if st.button("Effacer l'Entrée"):
            audio_data = None
            st.experimental_rerun()
    with cB:
        if st.button("Vider l'Historique"):
            st.session_state["history"]=[]
            save_history([])
            st.sidebar.success("Historique vidé.")

    final_aud = None
    if audio_data:
        try:
            seg = AudioSegment.from_file(io.BytesIO(audio_data))
            orig_sec = len(seg)/1000.0
            st.write(f"Durée Originale : {human_time(orig_sec)}")
            final_aud = seg
            if remove_sil:
                final_aud = remove_silences_classic(final_aud)
            if abs(speed_factor-1.0)>1e-2:
                final_aud = accelerate_ffmpeg(final_aud, speed_factor)
            final_sec = len(final_aud)/1000.0
            st.write(f"Durée Finale : {human_time(final_sec)}")
            bufIO = io.BytesIO()
            final_aud.export(bufIO, format="wav")
            st.write("### Aperçu Audio Transformé")
            st.audio(bufIO.getvalue(), format="audio/wav")
            if final_sec<orig_sec:
                st.success(f"Gagné {human_time(orig_sec-final_sec)}.")
        except Exception as e:
            st.error(f"Erreur Prétraitement : {e}")

    api_keys = []
    key_ids = []
    for k in st.secrets:
        if k.startswith("NOVA"):
            api_keys.append(st.secrets[k])
            key_ids.append(k)
    if not api_keys:
        st.sidebar.error("Pas de clé API config.")
        st.stop()

    if final_aud and st.button("Transcrire"):
        try:
            st.info("Traitement en cours...")
            dur_s = len(final_aud)/1000.0
            startT = time.time()

            from_aud = AudioSegment.from_file(io.BytesIO(audio_data))
            origin_sec = len(from_aud)/1000.0

            if accessibility:
                sel = select_api_keys(credits, key_ids, dur_s)
                if len(sel)<2:
                    st.error("Pas assez de clés API pour double.")
                    st.stop()
                kid1, c1 = sel[0]
                kid2, c2 = sel[1]
                with open("temp_in.wav","wb") as ff:
                    final_aud.export(ff, format="wav")
                t1 = nova_api.transcribe_audio("temp_in.wav", st.secrets[kid1], language_selection, "nova-2")
                st.write("1ère transcription terminée ? En attente de la seconde...")
                t2 = nova_api.transcribe_audio("temp_in.wav", st.secrets[kid2], language_selection, "whisper-large")
                elapsed = time.time()-startT
                if t1 and t2:
                    st.success(f"Double transcription terminée en {elapsed:.1f}s.")
                    cL,cR = st.columns(2)
                    with cL:
                        st.subheader("Résultat Nova 2")
                        st.text_area("Texte Nova 2", t1, height=180)
                        copy_to_clipboard(t1)
                    with cR:
                        st.subheader("Résultat Whisper Large")
                        st.text_area("Texte Whisper", t2, height=180)
                        copy_to_clipboard(t2)
                    gain=0
                    if dur_s<origin_sec:
                        gain = origin_sec-dur_s
                    st.write(f"Durée Finale : {human_time(dur_s)} (gagné {human_time(gain)}) | Temps={human_time(elapsed)} | Coût=~${c1+c2:.2f}")
                    alias1 = generate_alias(6) if not file_name else file_name+"_Nova2"
                    alias2 = generate_alias(6) if not file_name else file_name+"_Whisper"
                    e1 = {
                        "Alias/Nom": alias1,
                        "Méthode": "Nova 2",
                        "Modèle": "nova-2",
                        "Durée": human_time(origin_sec),
                        "Temps": human_time(elapsed),
                        "Coût": f"${c1:.2f}",
                        "Transcription": t1,
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Audio Binaire": audio_data.hex()
                    }
                    e2 = {
                        "Alias/Nom": alias2,
                        "Méthode": "Whisper Large",
                        "Modèle": "whisper-large",
                        "Durée": human_time(origin_sec),
                        "Temps": human_time(elapsed),
                        "Coût": f"${c2:.2f}",
                        "Transcription": t2,
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Audio Binaire": audio_data.hex()
                    }
                    st.session_state["history"].extend([e1,e2])
                    save_history(st.session_state["history"])
                    st.success("Historique mis à jour.")
                    credits[kid1] -= c1
                    credits[kid2] -= c2
                    save_credits(credits)
                if os.path.exists("temp_in.wav"):
                    os.remove("temp_in.wav")
            else:
                sel = select_api_keys(credits, key_ids, dur_s)
                if len(sel)<1:
                    st.error("Aucune clé API dispo.")
                    st.stop()
                kId, cst = sel[0]
                with open("temp_in.wav","wb") as ff:
                    final_aud.export(ff, format="wav")
                trans = nova_api.transcribe_audio("temp_in.wav", st.secrets[kId], language_selection, chosen_model)
                elapsed = time.time()-startT
                if trans:
                    st.success(f"Transcription terminée en {elapsed:.1f}s.")
                    st.text_area("Texte Transcrit", trans, height=180)
                    copy_to_clipboard(trans)
                    gain=0
                    if dur_s<origin_sec:
                        gain=origin_sec-dur_s
                    st.write(f"Durée Finale : {human_time(dur_s)} (gagné {human_time(gain)}) | Temps={human_time(elapsed)} | Coût=~${cst:.2f}")
                    aliasx = generate_alias(6) if not file_name else file_name
                    eX = {
                        "Alias/Nom": aliasx,
                        "Méthode": chosen_model,
                        "Modèle": chosen_model,
                        "Durée": human_time(origin_sec),
                        "Temps": human_time(elapsed),
                        "Coût": f"${cst:.2f}",
                        "Transcription": trans,
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Audio Binaire": audio_data.hex()
                    }
                    st.session_state["history"].append(eX)
                    save_history(st.session_state["history"])
                    st.success("Historique mis à jour.")
                    credits[kId] -= cst
                    save_credits(credits)
                if os.path.exists("temp_in.wav"):
                    os.remove("temp_in.wav")
        except Exception as e:
            st.error(f"Erreur Transcription : {e}")

    display_history()

def main_wrapper():
    main()

if __name__=="__main__":
    main_wrapper()
