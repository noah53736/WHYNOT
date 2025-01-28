# app.py

import streamlit as st
import os
import io
import random
import string
from datetime import datetime
from pydub import AudioSegment
import nova_api

###########################################
#       INITIALISATION ET UTILITAIRES     #
###########################################

def init_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "key_index" not in st.session_state:
        st.session_state["key_index"] = 0
    if "keys_failed" not in st.session_state:
        st.session_state["keys_failed"] = set()

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

###########################################
#       GESTION DES CLÉS API DEEPGRAM     #
###########################################

def get_api_keys():
    # Récupère toutes les clés API qui commencent par NOVA
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def get_valid_key(api_keys):
    """
    Retourne la première clé valide (non marquée comme échouée) 
    en faisant une rotation cyclique.
    """
    count = len(api_keys)
    tries = 0
    while tries < count:
        k_index = st.session_state["key_index"] % count
        candidate = api_keys[k_index]
        st.session_state["key_index"] += 1
        tries += 1
        if candidate not in st.session_state["keys_failed"]:
            return candidate
    return None

def mark_key_failed(k: str):
    st.session_state["keys_failed"].add(k)

###########################################
#       LOGIQUE PRINCIPALE DE L'APP       #
###########################################

def main():
    st.set_page_config(page_title="N-B-L Audio", layout="wide")
    st.title("N-B-L Audio : Transcription Grand Public (One Key at a Time)")

    init_state()
    history = st.session_state["history"]
    api_keys = get_api_keys()
    if not api_keys:
        st.error("Aucune clé API n'a été définie dans vos secrets Streamlit.")
        st.stop()

    # --- 1) Entrée Audio ---
    st.subheader("Entrée Audio")
    audio_data = []
    file_names = []

    input_col = st.columns([1])[0]
    input_type = input_col.radio(
        "Choisissez la source :", 
        ["Fichier (Upload)", "Micro (Enregistrement)"]
    )

    if input_type == "Fichier (Upload)":
        uploaded_files = input_col.file_uploader(
            "Importer un ou plusieurs fichiers audio", 
            type=["mp3","wav","m4a","ogg","webm"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for f in uploaded_files:
                audio_data.append(f.read())
                file_names.append(f.name)
                st.audio(f, format=f.type)
    else:
        max_mics = 4
        mic_count = input_col.number_input(
            "Nombre de microphones à utiliser", 
            min_value=1, 
            max_value=max_mics, 
            value=1,
            step=1
        )
        for i in range(mic_count):
            mic = input_col.audio_input(f"Microphone {i+1}")
            if mic:
                audio_data.append(mic.read())
                file_names.append(f"Micro_{i+1}")
                st.audio(mic, format=mic.type)

    st.write("---")

    # --- 2) Options Transcription ---
    st.subheader("Options de Transcription")
    colA, colB = st.columns([1,1])
    with colA:
        model_selection = st.radio(
            "Choisissez le modèle :",
            ["Nova 2", "Whisper Large"]
        )
    with colB:
        language_selection = "fr"
        if model_selection == "Whisper Large":
            language_selection = st.radio(
                "Langue pour Whisper :",
                ["fr", "en"]
            )

    double_trans = st.checkbox(
        "Double Transcription (Nova 2 + Whisper Large)",
        value=False
    )

    # --- 3) Bouton "Transcrire" ---
    if st.button("Transcrire") and audio_data:
        st.info("Début du traitement...")
        total_cost = 0
        progress_main = st.progress(0)
        total_files = len(audio_data)

        for idx, raw_data in enumerate(audio_data):
            st.write(f"**Fichier {idx+1}/{total_files}**")
            seg = AudioSegment.from_file(io.BytesIO(raw_data))
            orig_sec = len(seg)/1000.0

            # Nom du fichier
            f_name = file_names[idx] if idx < len(file_names) else f"Audio_{idx+1}"

            st.write(f"Durée : {human_time(orig_sec)}")

            # Segmentation si le fichier est volumineux ET si Whisper Large
            need_seg = (model_selection == "Whisper Large" and len(raw_data) > 20*1024*1024)
            chunks = [seg]
            if need_seg:
                st.info("Segmentation en morceaux de 20 secondes pour Whisper Large.")
                chunk_duration = 20
                chunks = [
                    seg[i*chunk_duration*1000 : (i+1)*chunk_duration*1000]
                    for i in range((len(seg)//(chunk_duration*1000))+1)
                ]

            # Exporter les segments
            chunk_files = []
            for c_i, chunk_seg in enumerate(chunks):
                temp_file = f"temp_chunk_{idx}_{c_i}.wav"
                chunk_seg.export(temp_file, format="wav")
                chunk_files.append(temp_file)

            # --- Transcription : Clé unique, si échec => on tente la suivante ---
            transcripts = []
            costs_list = []

            if double_trans:
                # On fait NOVA 2 + Whisper Large
                # 1) On récupère une clé pour Nova 2
                k1 = get_valid_key(api_keys)
                # 2) On récupère une clé pour Whisper Large
                k2 = get_valid_key(api_keys)
                if not k1 or not k2:
                    st.error("Plus de clé API valide disponible.")
                    break

                progress_chunks = st.progress(0)
                nb_chunks = len(chunk_files)*2
                done_chunks = 0

                for c_file in chunk_files:
                    # Transcription Nova 2
                    t_nova = nova_api.transcribe_audio(c_file, k1, "fr", "nova-2")
                    if t_nova == "":
                        # On marque la clé comme échouée si 401 ou trans vide
                        mark_key_failed(k1)
                        # On tente une autre clé
                        k1b = get_valid_key(api_keys)
                        if k1b:
                            t_nova = nova_api.transcribe_audio(c_file, k1b, "fr", "nova-2")
                            if t_nova == "":
                                mark_key_failed(k1b)
                        else:
                            st.error("Aucune clé valide pour Nova 2.")
                            t_nova = ""

                    # Transcription Whisper Large
                    t_whisper = nova_api.transcribe_audio(c_file, k2, language_selection, "whisper-large")
                    if t_whisper == "":
                        mark_key_failed(k2)
                        k2b = get_valid_key(api_keys)
                        if k2b:
                            t_whisper = nova_api.transcribe_audio(c_file, k2b, language_selection, "whisper-large")
                            if t_whisper == "":
                                mark_key_failed(k2b)
                        else:
                            st.error("Aucune clé valide pour Whisper Large.")
                            t_whisper = ""

                    transcripts.append({
                        "Nova 2": t_nova,
                        "Whisper Large": t_whisper
                    })
                    cost_chunk = len(AudioSegment.from_file(c_file))/1000.0 * 0.007
                    costs_list.extend([cost_chunk, cost_chunk])
                    done_chunks += 2
                    progress_chunks.progress(done_chunks/nb_chunks)
                    os.remove(c_file)

                # Unification
                full_nova = " ".join([t["Nova 2"] for t in transcripts])
                full_whisper = " ".join([t["Whisper Large"] for t in transcripts])

                st.success(f"Fichier {f_name} : Double Transcription Terminée")
                leftC, rightC = st.columns(2)
                with leftC:
                    st.subheader(f"NOVA 2 - {f_name}")
                    st.text_area("Texte Nova 2", full_nova, height=150)
                    copy_to_clipboard(full_nova)
                with rightC:
                    st.subheader(f"Whisper Large - {f_name}")
                    st.text_area("Texte Whisper", full_whisper, height=150)
                    copy_to_clipboard(full_whisper)

                # Historique
                cost_x = sum(costs_list)
                total_cost += cost_x
                alias1 = f"{f_name}_NOVA2"
                alias2 = f"{f_name}_WHISPER"
                e1 = {
                    "Alias/Nom": alias1,
                    "Méthode": "Nova 2",
                    "Modèle": "nova-2",
                    "Durée": human_time(orig_sec),
                    "Temps": human_time(len(seg)/1000.0),
                    "Coût": f"${cost_x/2:.2f}",
                    "Transcription": full_nova,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                e2 = {
                    "Alias/Nom": alias2,
                    "Méthode": "Whisper Large",
                    "Modèle": "whisper-large",
                    "Durée": human_time(orig_sec),
                    "Temps": human_time(len(seg)/1000.0),
                    "Coût": f"${cost_x/2:.2f}",
                    "Transcription": full_whisper,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].extend([e1,e2])

            else:
                # Transcription Simple
                k_simple = get_valid_key(api_keys)
                if not k_simple:
                    st.error("Aucune clé API valide pour la transcription simple.")
                    break
                progress_chunks = st.progress(0)
                nb_chunks = len(chunk_files)
                done_chunks = 0
                list_trans = []

                for c_file in chunk_files:
                    trans = nova_api.transcribe_audio(
                        c_file, 
                        k_simple,
                        language_selection if model_selection=="Whisper Large" else "fr",
                        "whisper-large" if model_selection=="Whisper Large" else "nova-2"
                    )
                    if trans == "":
                        mark_key_failed(k_simple)
                        k_bis = get_valid_key(api_keys)
                        if k_bis:
                            trans = nova_api.transcribe_audio(
                                c_file,
                                k_bis,
                                language_selection if model_selection=="Whisper Large" else "fr",
                                "whisper-large" if model_selection=="Whisper Large" else "nova-2"
                            )
                            if trans=="":
                                mark_key_failed(k_bis)
                                st.error("Impossible de transcrire avec une clé valide.")
                        else:
                            st.error("Aucune clé valide disponible.")
                            trans = ""
                    list_trans.append(trans)
                    cost_chunk = len(AudioSegment.from_file(c_file))/1000.0 * 0.007
                    costs_list.append(cost_chunk)
                    done_chunks += 1
                    progress_chunks.progress(done_chunks/nb_chunks)
                    os.remove(c_file)

                text_final = " ".join(list_trans)
                st.success(f"Fichier {f_name} : Transcription Terminée")
                st.text_area(f"Texte Transcrit - {f_name}", text_final, height=150)
                copy_to_clipboard(text_final)

                cost_simple = sum(costs_list)
                total_cost += cost_simple
                alias_simp = f"{f_name}_{model_selection}"
                e_simp = {
                    "Alias/Nom": alias_simp,
                    "Méthode": model_selection,
                    "Modèle": "whisper-large" if model_selection=="Whisper Large" else "nova-2",
                    "Durée": human_time(orig_sec),
                    "Temps": human_time(len(seg)/1000.0),
                    "Coût": f"${cost_simple:.2f}",
                    "Transcription": text_final,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Audio Binaire": raw_data.hex()
                }
                st.session_state["history"].append(e_simp)

        # Après tous les fichiers
        display_history(st.session_state["history"])
        st.write("---")
        st.write("### Aperçu Audio")
        for en in st.session_state["history"]:
            st.audio(bytes.fromhex(en["Audio Binaire"]), format="audio/wav")

def main_wrapper():
    try:
        main()
    except Exception as e:
        st.error(f"Erreur fatale : {e}")

if __name__ == "__main__":
    main_wrapper()
