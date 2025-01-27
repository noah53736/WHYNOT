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

# Limite du "chunk" si tu veux découper côté serveur (en MB) avant l'envoi final
CHUNK_LIMIT_MB = 180

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
    tmp_in = "temp_acc_in.wav"
    tmp_out = "temp_acc_out.wav"
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
    selected_keys = []
    for key_id in key_ids:
        if key_id in used_keys:
            continue
        if credits.get(key_id, 0) >= cost:
            selected_keys.append((key_id, cost))
            if len(selected_keys) == 2:
                break
    return selected_keys

def copy_to_clipboard(text):
    copy_button_html = f"""
        <script>
            function copyText(t) {{
                navigator.clipboard.writeText(t).then(function() {{
                    alert('Texte copié dans le presse-papiers!');
                }}, function(err) {{
                    alert('Échec de la copie : ' + err);
                }});
            }}
        </script>
        <button onclick="copyText(`{text}`)" style="padding: 5px 10px; margin-top: 5px;">Copier</button>
    """
    st.components.v1.html(copy_button_html)

def load_credits():
    if os.path.exists(CREDITS_FILE):
        with open(CREDITS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_credits(credits):
    with open(CREDITS_FILE, "w") as f:
        json.dump(credits, f, indent=4)

def display_history():
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    hist = st.session_state.get("history", [])
    if hist:
        table = []
        for en in hist:
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
        for en in reversed(hist[-3:]):
            st.sidebar.markdown(f"**{en['Alias/Nom']}** – {en['Date']}")
            ab = bytes.fromhex(en["Audio Binaire"])
            st.sidebar.audio(ab, format="audio/wav")
    else:
        st.sidebar.info("Historique vide.")

def main():
    st.set_page_config(page_title="N-B-L Audio", layout="wide")
    st.title("N-B-L Audio : Transcription Grand Public")

    if "history" not in st.session_state:
        st.session_state["history"] = load_history()
    history = st.session_state["history"]
    credits = load_credits()

    with st.sidebar:
        st.header("Options de Transcription")
        model_selection = st.selectbox("Choix Modèle", ["Nova 2", "Whisper Large"])
        model_mapping = {"Nova 2": "nova-2", "Whisper Large": "whisper-large"}
        selected_model = model_mapping.get(model_selection, "nova-2")
        language_selection = st.selectbox("Langue de Transcription", ["fr", "en"])
        accessibility = st.checkbox("Double Transcription (Accessibilité)", False)
        st.write("---")
        st.header("Transformations")
        remove_sil = st.checkbox("Supprimer Silences", False)
        speed_factor = st.slider("Accélération", 0.5, 4.0, 1.0, 0.1)

    # Récupérer les clés depuis st.secrets
    api_keys = []
    key_ids = []
    for k in st.secrets:
        if k.startswith("NOVA"):
            api_keys.append(st.secrets[k])
            key_ids.append(k)
    if not api_keys:
        st.sidebar.error("Aucune clé API disponible.")
        st.stop()

    st.write("## Choisissez la Source Audio")
    audio_data = None
    file_name = None

    input_choice = st.radio("Entrée :", ["Fichier (Upload)", "Micro (Enregistrement)"])
    if input_choice == "Fichier (Upload)":
        uploaded_file = st.file_uploader("Importez votre audio (mp3, wav, etc.)", type=["mp3","wav","m4a","ogg","webm"])
        if uploaded_file:
            if uploaded_file.size > 200 * 1024 * 1024:
                st.warning("Le fichier dépasse la limite de 200MB imposée par Streamlit.")
            else:
                audio_data = uploaded_file.read()
                st.audio(uploaded_file, format=uploaded_file.type)
                file_name = st.text_input("Nom du Fichier (Optionnel)")
    else:
        mic = st.audio_input("Enregistrer via Micro")
        if mic:
            audio_data = mic.read()
            st.audio(mic, format=mic.type)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Effacer l'Entrée"):
            audio_data = None
            st.experimental_rerun()
    with colB:
        if st.button("Vider l'Historique"):
            st.session_state["history"] = []
            save_history([])
            st.sidebar.success("Historique vidé.")

    final_aud = None
    if audio_data:
        try:
            orig_seg = AudioSegment.from_file(io.BytesIO(audio_data))
            original_sec = len(orig_seg)/1000.0
            st.write(f"Durée Originale : {human_time(original_sec)}")
            final_aud = orig_seg
            if remove_sil:
                final_aud = remove_silences_classic(final_aud)
            if abs(speed_factor-1.0)>1e-2:
                final_aud = accelerate_ffmpeg(final_aud, speed_factor)
            final_sec = len(final_aud)/1000.0
            st.write(f"Durée Finale : {human_time(final_sec)}")
            bufA = io.BytesIO()
            final_aud.export(bufA, format="wav")
            st.write("### Aperçu Audio Transformé")
            st.audio(bufA.getvalue(), format="audio/wav")
            if final_sec<original_sec:
                st.success(f"Gagné {human_time(original_sec-final_sec)} sur la durée audio.")
        except Exception as e:
            st.error(f"Erreur Pré-traitement : {e}")

    if final_aud and st.button("Lancer la Transcription"):
        try:
            st.info("Transcription en cours...")
            st.progress(0)
            duration_sec = len(final_aud)/1000.0
            cost_est = duration_sec*COST_PER_SEC
            start_t = time.time()
            # Sélection automatique d'une ou deux clés
            if accessibility:
                selected_keys = select_api_keys(credits, key_ids, duration_sec)
                if len(selected_keys)<2:
                    st.error("Pas assez de clés API pour la double transcription.")
                    st.stop()
                key1_id, c1 = selected_keys[0]
                key2_id, c2 = selected_keys[1]
                api1 = st.secrets[key1_id]
                api2 = st.secrets[key2_id]
                with open("temp_input.wav","wb") as f:
                    final_aud.export(f, format="wav")
                t1 = nova_api.transcribe_audio("temp_input.wav", api1, language_selection, "nova-2")
                st.progress(50)
                t2 = nova_api.transcribe_audio("temp_input.wav", api2, language_selection, "whisper-large")
                st.progress(100)
                elapsed = time.time()-start_t
                if t1 and t2:
                    st.success(f"Transcriptions terminées en {elapsed:.1f}s.")
                    colLeft, colRight = st.columns(2)
                    with colLeft:
                        st.subheader("Nova 2")
                        st.text_area("Transcription Nova 2", t1, height=200)
                        copy_to_clipboard(t1)
                    with colRight:
                        st.subheader("Whisper Large")
                        st.text_area("Transcription Whisper Large", t2, height=200)
                        copy_to_clipboard(t2)

                    gain_s = 0
                    if duration_sec<original_sec:
                        gain_s = original_sec-duration_sec
                    st.write(f"Durée Finale : {human_time(duration_sec)} (gagné {human_time(gain_s)}) | Temps: {human_time(elapsed)} | Coût ~${c1+c2:.2f}")
                    alias1 = generate_alias(6) if not file_name else f"{file_name}_Nova2"
                    alias2 = generate_alias(6) if not file_name else f"{file_name}_Whisper"
                    e1 = {
                        "Alias/Nom": alias1,
                        "Méthode": "Nova 2",
                        "Modèle": "nova-2",
                        "Durée": human_time(original_sec),
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
                        "Durée": human_time(original_sec),
                        "Temps": human_time(elapsed),
                        "Coût": f"${c2:.2f}",
                        "Transcription": t2,
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Audio Binaire": audio_data.hex()
                    }
                    history.extend([e1,e2])
                    st.session_state["history"] = history
                    save_history(history)
                    st.success("Historique mis à jour.")
                    credits[key1_id] -= c1
                    credits[key2_id] -= c2
                    save_credits(credits)
                if os.path.exists("temp_input.wav"):
                    os.remove("temp_input.wav")
            else:
                selected_keys = select_api_keys(credits, key_ids, duration_sec)
                if len(selected_keys)<1:
                    st.error("Pas de clé API disponible.")
                    st.stop()
                kid, costK = selected_keys[0]
                apik = st.secrets[kid]
                with open("temp_input.wav","wb") as f:
                    final_aud.export(f, format="wav")
                st.progress(50)
                transcription = nova_api.transcribe_audio("temp_input.wav", apik, language_selection, selected_model)
                st.progress(100)
                elapsed = time.time()-start_t
                if transcription:
                    st.success(f"Transcription terminée en {elapsed:.1f}s.")
                    st.text_area("Texte Transcrit", transcription, height=200)
                    copy_to_clipboard(transcription)
                    gain_s = 0
                    if duration_sec<original_sec:
                        gain_s = original_sec-duration_sec
                    st.write(f"Durée Finale : {human_time(duration_sec)} (gagné {human_time(gain_s)}) | Temps: {human_time(elapsed)} | Coût ~${costK:.2f}")
                    alias = generate_alias(6) if not file_name else file_name
                    e = {
                        "Alias/Nom": alias,
                        "Méthode": selected_model,
                        "Modèle": selected_model,
                        "Durée": human_time(original_sec),
                        "Temps": human_time(elapsed),
                        "Coût": f"${costK:.2f}",
                        "Transcription": transcription,
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Audio Binaire": audio_data.hex()
                    }
                    history.append(e)
                    st.session_state["history"] = history
                    save_history(history)
                    st.success("Historique mis à jour.")
                    credits[kid] -= costK
                    save_credits(credits)
                if os.path.exists("temp_input.wav"):
                    os.remove("temp_input.wav")
        except Exception as err:
            st.error(f"Erreur Transcription : {err}")

    display_history()

def main_wrapper():
    st.set_page_config(page_title="N-B-L Audio", layout="wide")
    main()

if __name__=="__main__":
    main_wrapper()
