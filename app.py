import streamlit as st
import os
import io
import time
import random
import string
from datetime import datetime
import json
import subprocess
from pydub import AudioSegment, silence
import nova_api

HISTORY_FILE = "historique.json"
CREDITS_FILE = "credits.json"
MIN_SILENCE_LEN = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
COST_PER_SEC = 0.007

def init_history():
    if not os.path.exists(HISTORY_FILE):
        history = []
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        init_history()
    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

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
    cmd = [
        "ffmpeg","-y","-i", tmp_in,
        "-filter:a", f_str,
        tmp_out
    ]
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
                    alert('Transcription copiée dans le presse-papiers!');
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
            credits = json.load(f)
    else:
        credits = {}
    return credits

def save_credits(credits):
    with open(CREDITS_FILE, "w") as f:
        json.dump(credits, f, indent=4)

def display_history():
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    history = st.session_state.get("history", [])
    if history:
        history_table = [
            {
                "Alias/Nom": entry["Alias/Nom"],
                "Méthode": entry["Méthode"],
                "Modèle": entry["Modèle"],
                "Durée": entry["Durée"],
                "Temps": entry["Temps"],
                "Coût": entry["Coût"],
                "Date": entry["Date"]
            }
            for entry in history
        ]
        st.sidebar.table(history_table[::-1])
        st.sidebar.write("### Aperçus Audio")
        for entry in reversed(history[-3:]):
            st.sidebar.markdown(f"**{entry['Alias/Nom']}** – {entry['Date']}")
            audio_bytes = bytes.fromhex(entry["Audio Binaire"])
            st.sidebar.audio(audio_bytes, format="audio/wav")
    else:
        st.sidebar.info("Historique vide.")

def main():
    st.set_page_config(page_title="NBL Audio – DeepGram", layout="wide")
    st.title("NBL Audio – DeepGram")
    if "history" not in st.session_state:
        st.session_state["history"] = load_history()
    history = st.session_state["history"]
    credits = load_credits()

    with st.sidebar:
        st.header("Options Transcription")
        model_selection = st.selectbox("Sélectionner le Modèle de Transcription", ["Nova 2", "Whisper Large"])
        model_mapping = {"Nova 2": "nova-2", "Whisper Large": "whisper-large"}
        selected_model = model_mapping.get(model_selection, "nova-2")
        language_selection = st.selectbox("Sélectionner la Langue", ["fr", "en"])
        accessibility = st.checkbox("Activer l'accessibilité (Double Transcription)", False)
        st.write("---")
        st.header("Transformations")
        remove_sil = st.checkbox("Supprimer silences", False)
        speed_factor = st.slider("Accélération (time-stretch)", 0.5, 4.0, 1.0, 0.1)

    api_keys = []
    key_ids = []
    for key in st.secrets:
        if key.startswith("NOVA"):
            api_keys.append(st.secrets[key])
            key_ids.append(key)
    if not api_keys:
        st.sidebar.error("Aucune clé API DeepGram trouvée dans les secrets.")
        st.stop()

    st.write("## Source Audio")
    audio_data = None
    file_name = None
    input_choice = st.radio("Fichier ou Micro ?", ["Fichier", "Micro"])
    if input_choice == "Fichier":
        uploaded_file = st.file_uploader("Fichier audio (mp3, wav, m4a, ogg, webm)", type=["mp3","wav","m4a","ogg","webm"])
        if uploaded_file:
            if uploaded_file.size > 100 * 1024 * 1024:
                st.error("Le fichier est trop volumineux. Max 100MB.")
            else:
                audio_data = uploaded_file.read()
                st.audio(uploaded_file, format=uploaded_file.type)
                file_name = st.text_input("Renommer le fichier (optionnel)")
    else:
        mic_input = st.audio_input("Enregistrement micro")
        if mic_input:
            audio_data = mic_input.read()
            st.audio(mic_input, format=mic_input.type)

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Clear Uploaded File"):
            audio_data = None
            st.experimental_rerun()
    with colB:
        if st.button("Clear History"):
            history = []
            st.session_state["history"] = history
            save_history(history)
            st.sidebar.success("Historique vidé.")

    final_aud = None
    original_sec = 0.0
    if audio_data:
        try:
            aud = AudioSegment.from_file(io.BytesIO(audio_data))
            original_sec = len(aud) / 1000.0
            st.write(f"Durée d'origine : {human_time(original_sec)}")
            final_aud = aud
            if remove_sil:
                final_aud = remove_silences_classic(final_aud)
            if abs(speed_factor - 1.0) > 1e-2:
                final_aud = accelerate_ffmpeg(final_aud, speed_factor)
            final_sec = len(final_aud) / 1000.0
            st.write(f"Durée finale : {human_time(final_sec)}")
            bufp = io.BytesIO()
            final_aud.export(bufp, format="wav")
            st.write("### Aperçu Audio transformé")
            st.audio(bufp.getvalue(), format="audio/wav")
            if final_sec < original_sec:
                gain_sec = original_sec - final_sec
                st.success(f"Gagné {human_time(gain_sec)} grâce aux transformations.")
        except Exception as e:
            st.error(f"Erreur prétraitement : {e}")

    if final_aud and st.button("Transcrire"):
        try:
            st.write("Transcription en cours...")
            duration_sec = len(final_aud) / 1000.0
            cost = duration_sec * COST_PER_SEC
            start_t = time.time()

            if accessibility:
                selected_keys = select_api_keys(credits, key_ids, duration_sec)
                if len(selected_keys) < 2:
                    st.error("Pas assez de clés API disponibles.")
                    st.stop()
                key1_id, cost1 = selected_keys[0]
                key2_id, cost2 = selected_keys[1]
                api1 = st.secrets[key1_id]
                api2 = st.secrets[key2_id]
                with open("temp_in.wav","wb") as ff:
                    final_aud.export(ff, format="wav")
                tr1 = nova_api.transcribe_audio("temp_in.wav", api1, language_selection, "nova-2")
                tr2 = nova_api.transcribe_audio("temp_in.wav", api2, language_selection, "whisper-large")
                elapsed = time.time() - start_t
                if tr1 and tr2:
                    st.success(f"Transcription terminée en {elapsed:.2f}s")
                    st.subheader("Nova 2")
                    st.text_area("Transcription Nova 2", tr1, height=150)
                    copy_to_clipboard(tr1)
                    st.subheader("Whisper Large")
                    st.text_area("Transcription Whisper Large", tr2, height=150)
                    copy_to_clipboard(tr2)
                    gain = 0
                    if duration_sec < original_sec:
                        gain = original_sec - duration_sec
                    st.write(f"Durée finale : {human_time(duration_sec)} (gagné {human_time(gain)}) | Temps : {human_time(elapsed)} | Coût=${cost1+cost2:.2f}")
                    alias1 = generate_alias(6) if not file_name else file_name+"_Nova2"
                    alias2 = generate_alias(6) if not file_name else file_name+"_WhisperLarge"
                    e1 = {
                        "Alias/Nom": alias1,
                        "Méthode": "Nova 2",
                        "Modèle": "nova-2",
                        "Durée": human_time(original_sec),
                        "Temps": human_time(elapsed),
                        "Coût": f"${cost1:.2f}",
                        "Transcription": tr1,
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Audio Binaire": audio_data.hex()
                    }
                    e2 = {
                        "Alias/Nom": alias2,
                        "Méthode": "Whisper Large",
                        "Modèle": "whisper-large",
                        "Durée": human_time(original_sec),
                        "Temps": human_time(elapsed),
                        "Coût": f"${cost2:.2f}",
                        "Transcription": tr2,
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Audio Binaire": audio_data.hex()
                    }
                    history.extend([e1,e2])
                    st.session_state["history"] = history
                    save_history(history)
                    st.info("Historique mis à jour.")
                    credits[key1_id] -= cost1
                    credits[key2_id] -= cost2
                    save_credits(credits)
                if os.path.exists("temp_in.wav"):
                    os.remove("temp_in.wav")
            else:
                selected_keys = select_api_keys(credits, key_ids, duration_sec)
                if len(selected_keys) < 1:
                    st.error("Clés API insuffisantes.")
                    st.stop()
                key_id, cst = selected_keys[0]
                apik = st.secrets[key_id]
                with open("temp_in.wav","wb") as ff:
                    final_aud.export(ff, format="wav")
                tr = nova_api.transcribe_audio("temp_in.wav", apik, language_selection, selected_model)
                elapsed = time.time() - start_t
                if tr:
                    st.success(f"Transcription terminée en {elapsed:.2f}s")
                    st.text_area("Transcription", tr, height=150)
                    copy_to_clipboard(tr)
                    gain = 0
                    if duration_sec < original_sec:
                        gain = original_sec - duration_sec
                    st.write(f"Durée finale : {human_time(duration_sec)} (gagné {human_time(gain)}) | Temps : {human_time(elapsed)} | Coût=${cst:.2f}")
                    alias = generate_alias(6) if not file_name else file_name
                    e = {
                        "Alias/Nom": alias,
                        "Méthode": selected_model,
                        "Modèle": selected_model,
                        "Durée": human_time(original_sec),
                        "Temps": human_time(elapsed),
                        "Coût": f"${cst:.2f}",
                        "Transcription": tr,
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Audio Binaire": audio_data.hex()
                    }
                    history.append(e)
                    st.session_state["history"] = history
                    save_history(history)
                    st.info("Historique mis à jour.")
                    credits[key_id] -= cst
                    save_credits(credits)
                if os.path.exists("temp_in.wav"):
                    os.remove("temp_in.wav")
        except Exception as e:
            st.error(f"Erreur transcription : {e}")

    display_history()

def display_history():
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    hist = st.session_state.get("history", [])
    if hist:
        table = [
            {
                "Alias/Nom": en["Alias/Nom"],
                "Méthode": en["Méthode"],
                "Modèle": en["Modèle"],
                "Durée": en["Durée"],
                "Temps": en["Temps"],
                "Coût": en["Coût"],
                "Date": en["Date"]
            }
            for en in hist
        ]
        st.sidebar.table(table[::-1])
        st.sidebar.write("### Aperçus Audio")
        for en in reversed(hist[-3:]):
            st.sidebar.markdown(f"**{en['Alias/Nom']}** – {en['Date']}")
            ab = bytes.fromhex(en["Audio Binaire"])
            st.sidebar.audio(ab, format="audio/wav")
    else:
        st.sidebar.info("Historique vide.")

def load_credits():
    if os.path.exists(CREDITS_FILE):
        with open(CREDITS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_credits(credits):
    with open(CREDITS_FILE, "w") as f:
        json.dump(credits, f, indent=4)

def main_wrapper():
    main()

if __name__ == "__main__":
    main_wrapper()
