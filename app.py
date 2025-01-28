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

# Initialisation de l'historique et de l'index des clés API dans le State de Streamlit
def init_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "key_index" not in st.session_state:
        st.session_state["key_index"] = 0

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
    # Récupérer les clés API DeepGram à partir des secrets
    return [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]

def get_next_key(api_keys):
    key_count = len(api_keys)
    if key_count == 0:
        st.error("Aucune clé API disponible.")
        st.stop()
    key = api_keys[st.session_state["key_index"]]
    st.session_state["key_index"] = (st.session_state["key_index"] + 1) % key_count
    return key

def main():
    st.set_page_config(page_title="N-B-L Audio", layout="wide")
    st.title("N-B-L Audio : Transcription Grand Public")

    init_state()
    history = st.session_state["history"]

    st.write("## Entrée Audio")
    audio_data = []
    file_names = []

    # Choix du type d'entrée
    input_type = st.radio("Type d'Entrée :", ["Fichier (Upload)", "Micro (Enregistrement)"], key="input_type")

    if input_type == "Fichier (Upload)":
        upf = st.file_uploader("Importer l'audio", type=["mp3","wav","m4a","ogg","webm"], accept_multiple_files=True, key="file_uploader")
        if upf:
            if any(file.size > 300*1024*1024 for file in upf):
                st.warning("Un ou plusieurs fichiers dépassent 300MB.")
            else:
                for file in upf:
                    audio_data.append(file.read())
                    file_names.append(file.name)
                    st.audio(file, format=file.type)
    else:
        # Choix du nombre de microphones (limité à 4)
        max_mics = 4
        num_mics = st.number_input("Nombre de microphones à utiliser :", min_value=1, max_value=max_mics, step=1, key="num_mics")
        for i in range(int(num_mics)):
            mic_input = st.audio_input(f"Microphone {i+1}", key=f"audio_input_{i}")
            if mic_input:
                audio_data.append(mic_input.read())
                file_names.append(f"Microphone {i+1}")
                st.audio(mic_input, format=mic_input.type)

    st.write("---")
    st.write("## Options de Transcription")
    col1, col2 = st.columns([1,1])
    with col1:
        model_selection = st.selectbox("Modèle :", ["Nova 2", "Whisper Large"], key="model_selection")
    with col2:
        if model_selection == "Whisper Large":
            language_selection = st.selectbox("Langue :", ["fr", "en"], key="language_selection")
        else:
            language_selection = "fr"  # Valeur par défaut ou ignorée

    accessibility = st.checkbox("Double Transcription (Nova 2 et Whisper Large)", key="double_transcription")

    # Bouton pour lancer la transcription
    if st.button("Transcrire") and audio_data:
        try:
            st.info("Traitement en cours...")
            transcriptions = []
            total_cost = 0
            api_keys = get_api_keys()
            if len(api_keys) == 0:
                st.error("Aucune clé API disponible. Veuillez vérifier vos secrets.")
                st.stop()

            for idx, data in enumerate(audio_data):
                seg = AudioSegment.from_file(io.BytesIO(data))
                orig_sec = len(seg)/1000.0
                st.write(f"**Fichier {idx+1} : {file_names[idx]}** - Durée Originale : {human_time(orig_sec)}")

                # Déterminer si la segmentation est nécessaire
                needs_segmentation = False
                if model_selection == "Whisper Large" and len(data) > 20*1024*1024:
                    needs_segmentation = True

                if needs_segmentation:
                    st.info("Fichier trop volumineux, segmentation en morceaux de 20 secondes.")
                    # Segmentation en 20 secondes
                    chunk_duration = 20  # secondes
                    chunks = [seg[i*chunk_duration*1000:(i+1)*chunk_duration*1000] for i in range((len(seg) // (chunk_duration*1000)) + 1)]
                else:
                    chunks = [seg]

                # Exporter les segments en fichiers temporaires
                chunk_files = []
                for i, chunk in enumerate(chunks):
                    temp_filename = f"temp_chunk_{idx}_{i}.wav"
                    chunk.export(temp_filename, format="wav")
                    chunk_files.append(temp_filename)

                # Sélection des clés API avec rotation
                if accessibility:
                    key1 = get_next_key(api_keys)
                    key2 = get_next_key(api_keys)
                else:
                    key = get_next_key(api_keys)

                # Transcription des chunks avec barre de progression
                transcripts = []
                costs = []
                total_chunks = len(chunk_files) * (2 if accessibility else 1)
                current_chunk = 0
                progress_bar = st.progress(0)

                for chunk_file in chunk_files:
                    if accessibility:
                        # Transcription avec Nova 2
                        trans1 = nova_api.transcribe_audio(chunk_file, key1, "fr", "nova-2")
                        if not trans1:
                            # Passer à la seconde clé si la première échoue
                            key1 = get_next_key(api_keys)
                            trans1 = nova_api.transcribe_audio(chunk_file, key1, "fr", "nova-2")
                        # Transcription avec Whisper Large
                        trans2 = nova_api.transcribe_audio(chunk_file, key2, language_selection, "whisper-large")
                        if not trans2:
                            # Passer à la seconde clé si la première échoue
                            key2 = get_next_key(api_keys)
                            trans2 = nova_api.transcribe_audio(chunk_file, key2, language_selection, "whisper-large")
                        transcripts.append({"Nova 2": trans1, "Whisper Large": trans2})
                        cost1 = len(AudioSegment.from_file(chunk_file))/1000.0 * 0.007
                        cost2 = len(AudioSegment.from_file(chunk_file))/1000.0 * 0.007
                        costs.extend([cost1, cost2])
                        current_chunk += 2
                        progress_bar.progress(current_chunk / total_chunks)
                    else:
                        # Transcription unique
                        trans = nova_api.transcribe_audio(chunk_file, key, language_selection if model_selection == "Whisper Large" else "fr", "whisper-large" if model_selection == "Whisper Large" else "nova-2")
                        transcripts.append({model_selection: trans})
                        cost = len(AudioSegment.from_file(chunk_file))/1000.0 * 0.007
                        costs.append(cost)
                        current_chunk += 1
                        progress_bar.progress(current_chunk / total_chunks)
                    os.remove(chunk_file)

                # Unification des transcriptions
                if accessibility:
                    full_trans_nova = " ".join([t["Nova 2"] for t in transcripts])
                    full_trans_whisper = " ".join([t["Whisper Large"] for t in transcripts])
                    transcriptions.append({"Nova 2": full_trans_nova, "Whisper Large": full_trans_whisper})
                else:
                    if model_selection == "Whisper Large":
                        full_trans = " ".join([t["Whisper Large"] for t in transcripts])
                        transcriptions.append({model_selection: full_trans})
                    else:
                        full_trans = " ".join([t["Nova 2"] for t in transcripts])
                        transcriptions.append({model_selection: full_trans})

                total_cost += sum(costs)

                # Affichage des résultats immédiatement après chaque transcription
                for trans in transcriptions:
                    if accessibility:
                        st.success(f"Double transcription du Fichier {idx+1} terminée.")
                        cL, cR = st.columns(2)
                        with cL:
                            st.subheader("Résultat Nova 2")
                            st.text_area(f"Texte Nova 2 - Fichier {idx+1}", trans["Nova 2"], height=180)
                            copy_to_clipboard(trans["Nova 2"])
                        with cR:
                            st.subheader("Résultat Whisper Large")
                            st.text_area(f"Texte Whisper Large - Fichier {idx+1}", trans["Whisper Large"], height=180)
                            copy_to_clipboard(trans["Whisper Large"])

                        alias1 = generate_alias(6) if not file_names[idx] else f"{file_names[idx]}_Nova2_{idx+1}"
                        alias2 = generate_alias(6) if not file_names[idx] else f"{file_names[idx]}_Whisper_{idx+1}"
                        entry1 = {
                            "Alias/Nom": alias1,
                            "Méthode": "Nova 2",
                            "Modèle": "nova-2",
                            "Durée": human_time(orig_sec),
                            "Temps": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Coût": f"${costs[2*idx]:.2f}",
                            "Transcription": trans["Nova 2"],
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Audio Binaire": data.hex()
                        }
                        entry2 = {
                            "Alias/Nom": alias2,
                            "Méthode": "Whisper Large",
                            "Modèle": "whisper-large",
                            "Durée": human_time(orig_sec),
                            "Temps": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Coût": f"${costs[2*idx+1]:.2f}",
                            "Transcription": trans["Whisper Large"],
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Audio Binaire": data.hex()
                        }
                        st.session_state["history"].extend([entry1, entry2])
                    else:
                        st.success(f"Transcription du Fichier {idx+1} terminée.")
                        st.text_area(f"Texte Transcrit - Fichier {idx+1}", trans[model_selection], height=180)
                        copy_to_clipboard(trans[model_selection])

                        alias = generate_alias(6) if not file_names[idx] else f"{file_names[idx]}_{model_selection}_{idx+1}"
                        entry = {
                            "Alias/Nom": alias,
                            "Méthode": model_selection,
                            "Modèle": "whisper-large" if model_selection == "Whisper Large" else "nova-2",
                            "Durée": human_time(orig_sec),
                            "Temps": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Coût": f"${costs[idx]:.2f}",
                            "Transcription": trans[model_selection],
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Audio Binaire": data.hex()
                        }
                        st.session_state["history"].append(entry)

        except Exception as e:
            st.error(f"Erreur lors de la transcription : {e}")

    # Affichage de l'historique et de l'aperçu audio
        display_history(st.session_state["history"])

        st.write("---")
        st.write("### Aperçu Audio")
        for en in st.session_state["history"]:
            st.audio(bytes.fromhex(en["Audio Binaire"]), format="audio/wav")

    if __name__ == "__main__":
        main()
