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

def init_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Index de clé (on ne segmente pas, on ne tourne pas de multiples fois)
    if "dg_key_index" not in st.session_state:
        st.session_state["dg_key_index"] = 0

def get_dg_keys():
    """ Récupère toutes les clés API DeepGram depuis secrets. """
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def pick_key(keys):
    """ Simple : on prend la clé courante, pas de rotation multiple. """
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
    if sec < 60:
        return f"{sec}s"
    elif sec < 3600:
        m, s = divmod(sec, 60)
        return f"{m}m{s}s"
    else:
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        return f"{h}h{m}m{s}s"

def copy_to_clipboard(txt):
    script_html = f"""
    <script>
        function copyText(t) {{
            navigator.clipboard.writeText(t).then(function() {{
                alert('Transcription copiée !');
            }}, function(err) {{
                alert('Échec : ' + err);
            }});
        }}
    </script>
    <button onclick="copyText(`{txt}`)" style="margin:5px;">Copier</button>
    """
    st.components.v1.html(script_html)

def display_history():
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    h = st.session_state["history"]
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
    else:
        st.sidebar.info("Historique vide.")

############################################
def main():
    st.set_page_config(page_title="NBL Audio", layout="wide")
    st.title("NBL Audio")

    init_state()
    dg_keys = get_dg_keys()
    if not dg_keys:
        st.error("Aucune clé DeepGram n'est configurée.")
        st.stop()

    history = st.session_state["history"]

    st.subheader("Entrée Audio")
    audio_data_list = []
    file_names = []

    input_type = st.radio("Source audio", ["Fichier (Upload)","Micro (Enregistrement)"], index=0)
    if input_type=="Fichier (Upload)":
        uploaded_files = st.file_uploader("Fichiers audio", type=["mp3","wav","m4a","ogg","webm"], accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                audio_data_list.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        nmics = st.number_input("Nombre de micros", min_value=1, max_value=4, value=1)
        for i in range(nmics):
            mic = st.audio_input(f"Micro {i+1}")
            if mic:
                audio_data_list.append(mic.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic, format=mic.type)

    st.write("---")
    st.subheader("Options de Transcription")
    colA, colB = st.columns([1,1])
    with colA:
        model_main = st.selectbox("Modèle Principal :", ["Nova 2","Whisper Large"])
    with colB:
        lang_main = "fr"
        if model_main=="Whisper Large":
            lang_main = st.selectbox("Langue (Whisper) :", ["fr","en"])

    double_trans = st.checkbox("Double Transcription (Nova 2 puis Whisper)", value=False)

    if st.button("Transcrire") and audio_data_list:
        st.info("Transcription en cours ...")
        for idx, raw in enumerate(audio_data_list):
            seg = AudioSegment.from_file(io.BytesIO(raw))
            dur_s = len(seg)/1000.0
            f_name = file_names[idx] if idx<len(file_names) else f"Fichier_{idx+1}"

            st.write(f"### Fichier {idx+1}: {f_name}")
            st.write(f"Durée : {human_time(dur_s)}")

            temp_wav = f"temp_input_{idx}.wav"
            seg.export(temp_wav, format="wav")

            if double_trans:
                # 1) Nova 2
                start_nova = time.time()
                dg_key_nova = pick_key(dg_keys)
                txt_nova = nova_api.transcribe_audio(temp_wav, dg_key_nova, "fr", "nova-2")
                end_nova = time.time()

                cA, cB = st.columns(2)
                with cA:
                    st.subheader("Résultat Nova 2")
                    st.text_area(f"Nova 2 - {f_name}", txt_nova, height=150)
                    copy_to_clipboard(txt_nova)

                eN = {
                    "Alias/Nom": f"{f_name}_NOVA2",
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": human_time(dur_s),
                    "Temps": f"{end_nova - start_nova:.1f}s",
                    "Coût": "?",
                    "Transcription": txt_nova,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw.hex()
                }
                st.session_state["history"].append(eN)

                # 2) Whisper Large
                start_whisper = time.time()
                dg_key_whisper = pick_key(dg_keys)
                txt_whisp = nova_api.transcribe_audio(temp_wav, dg_key_whisper, lang_main, "whisper-large")
                end_whisper = time.time()

                with cB:
                    st.subheader("Résultat Whisper Large")
                    st.text_area(f"Whisper - {f_name}", txt_whisp, height=150)
                    copy_to_clipboard(txt_whisp)

                eW = {
                    "Alias/Nom": f"{f_name}_WHISPER",
                    "Méthode": "Whisper Large",
                    "Modèle": "whisper-large",
                    "Durée": human_time(dur_s),
                    "Temps": f"{end_whisper - start_whisper:.1f}s",
                    "Coût": "?",
                    "Transcription": txt_whisp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw.hex()
                }
                st.session_state["history"].append(eW)
            else:
                # Simple
                start_t = time.time()
                dg_key_simple = pick_key(dg_keys)
                chosen_model = "nova-2" if model_main=="Nova 2" else "whisper-large"
                chosen_lang = "fr" if chosen_model=="nova-2" else lang_main
                txt_single = nova_api.transcribe_audio(temp_wav, dg_key_simple, chosen_lang, chosen_model)
                end_t = time.time()

                cA, _ = st.columns([1,1])
                with cA:
                    st.subheader(f"Résultat {model_main}")
                    st.text_area(f"{f_name}", txt_single, height=150)
                    copy_to_clipboard(txt_single)

                eS = {
                    "Alias/Nom": f"{f_name}_{chosen_model}",
                    "Méthode": model_main,
                    "Modèle": chosen_model,
                    "Durée": human_time(dur_s),
                    "Temps": f"{end_t - start_t:.1f}s",
                    "Coût": "?",
                    "Transcription": txt_single,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw.hex()
                }
                st.session_state["history"].append(eS)

            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        display_history()
        st.write("---")
        st.write("### Aperçu Audio")
        for en in st.session_state["history"]:
            st.audio(bytes.fromhex(en["Audio Binaire"]), format="audio/wav")

def main_wrapper():
    try:
        main()
    except Exception as e:
        st.error(f"Erreur Principale : {e}")

if __name__=="__main__":
    main_wrapper()
