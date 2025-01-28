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
    if "dg_key_index" not in st.session_state:
        st.session_state["dg_key_index"] = 0

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

def display_history(history):
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    if history:
        table = []
        for en in history:
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

def get_api_keys():
    """Récupérer toutes les clés API DeepGram depuis st.secrets (préfixées par 'NOVA')."""
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def pick_key(api_keys):
    """
    On prend la clé courante (dg_key_index),
    sans rotation multiple, on réessaie rien : 
    si ça foire => c'est la vie.
    """
    if not api_keys:
        return None
    idx = st.session_state["dg_key_index"]
    if idx>=len(api_keys):
        st.session_state["dg_key_index"] = 0
        idx=0
    return api_keys[idx]

def main():
    st.set_page_config(page_title="NBL Audio", layout="wide")
    st.title("NBL Audio")

    init_state()
    history = st.session_state["history"]
    keys = get_api_keys()

    if not keys:
        st.error("Aucune clé API n'a été configurée dans st.secrets.")
        st.stop()

    st.subheader("Entrée Audio")
    audio_data = []
    file_names = []

    # 1. Choix de la source
    source_type = st.radio("Source Audio", ["Fichier (Upload)", "Micro (Enregistrement)"])
    if source_type=="Fichier (Upload)":
        upf = st.file_uploader("Fichiers audio", 
            type=["mp3","wav","m4a","ogg","webm"],
            accept_multiple_files=True
        )
        if upf:
            for f in upf:
                audio_data.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        max_mics = 4
        nm = st.number_input("Nombre de micros", min_value=1, max_value=max_mics, value=1)
        for i in range(nm):
            mic = st.audio_input(f"Micro {i+1}")
            if mic:
                audio_data.append(mic.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic, format=mic.type)

    st.write("---")
    # 2. Options
    st.subheader("Options Transcription")
    colA, colB = st.columns([1,1])
    with colA:
        model_main = st.selectbox("Modèle Principal :", ["Nova 2","Whisper Large"])
    with colB:
        lang_main = "fr"
        if model_main=="Whisper Large":
            lang_main = st.selectbox("Langue pour Whisper", ["fr","en"])

    double_trans = st.checkbox("Double Transcription (Nova 2 -> Whisper)", value=False)

    # 3. Bouton Transcrire
    if st.button("Transcrire") and audio_data:
        st.info("Début de la transcription...")

        for idx, raw_audio in enumerate(audio_data):
            st.write(f"### Fichier {idx+1}")
            f_name = file_names[idx] if idx<len(file_names) else f"F_{idx+1}"
            
            # On crée un WAV simple, pas de segmentation
            temp_in = f"temp_input_{idx}.wav"
            seg = AudioSegment.from_file(io.BytesIO(raw_audio))
            seg.export(temp_in, format="wav")
            orig_sec = len(seg)/1000.0

            if double_trans:
                # 1) Nova 2
                st.write(f"**{f_name} => Nova 2**")
                start_nova = time.time()
                dg_key = pick_key(keys)
                txt_nova = nova_api.transcribe_audio(temp_in, dg_key, "fr", "nova-2")
                end_nova = time.time()
                with st.container():
                    leftC, rightC = st.columns(2)
                    with leftC:
                        st.subheader("NOVA 2")
                        st.text_area(f"Résultat Nova 2 - {f_name}", txt_nova, height=120)
                        copy_to_clipboard(txt_nova)

                eNova = {
                    "Alias/Nom": f"{f_name}_Nova2",
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": human_time(orig_sec),
                    "Temps": f"{end_nova - start_nova:.1f}s",
                    "Coût": "?",
                    "Transcription": txt_nova,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_audio.hex()
                }
                st.session_state["history"].append(eNova)

                # 2) Whisper
                st.write(f"**{f_name} => Whisper Large**")
                start_whisp = time.time()
                dg_key2 = pick_key(keys)
                txt_whisp = nova_api.transcribe_audio(temp_in, dg_key2, lang_main, "whisper-large")
                end_whisp = time.time()

                with st.container():
                    # on récupère la même colonne rightC
                    # pour avoir un affichage côte à côte
                    leftC, rightC = st.columns(2)
                    with rightC:
                        st.subheader("WHISPER LARGE")
                        st.text_area(f"Résultat Whisper - {f_name}", txt_whisp, height=120)
                        copy_to_clipboard(txt_whisp)

                eWhisp = {
                    "Alias/Nom": f"{f_name}_Whisper",
                    "Méthode": "Whisper Large",
                    "Modèle": "whisper-large",
                    "Durée": human_time(orig_sec),
                    "Temps": f"{end_whisp - start_whisp:.1f}s",
                    "Coût": "?",
                    "Transcription": txt_whisp,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_audio.hex()
                }
                st.session_state["history"].append(eWhisp)

            else:
                # Simple
                start_t = time.time()
                d_key = pick_key(keys)
                chosen_model = "nova-2" if model_main=="Nova 2" else "whisper-large"
                txt_single = nova_api.transcribe_audio(temp_in, d_key, lang_main if chosen_model=="whisper-large" else "fr", chosen_model)
                end_t = time.time()

                leftC, _ = st.columns([1,1])
                with leftC:
                    st.subheader(f"{model_main}")
                    st.text_area(f"Résultat - {f_name}", txt_single, height=120)
                    copy_to_clipboard(txt_single)

                eEntry = {
                    "Alias/Nom": f"{f_name}_{chosen_model}",
                    "Méthode": model_main,
                    "Modèle": chosen_model,
                    "Durée": human_time(orig_sec),
                    "Temps": f"{end_t - start_t:.1f}s",
                    "Coût": "?",
                    "Transcription": txt_single,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_audio.hex()
                }
                st.session_state["history"].append(eEntry)

            if os.path.exists(temp_in):
                os.remove(temp_in)

        display_history(st.session_state["history"])
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
