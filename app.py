# app.py
import streamlit as st
import os
import io
import time
import random
import string
from datetime import datetime
from pydub import AudioSegment
import nova_api

# Initialisation de l'historique dans le State de Streamlit
def init_history():
    if "history" not in st.session_state:
        st.session_state["history"] = []

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

def main():
    st.set_page_config(page_title="N-B-L Audio", layout="wide")
    st.title("N-B-L Audio : Transcription Grand Public")

    init_history()
    history = st.session_state["history"]

    st.write("## Entrée Audio")
    audio_data = []
    file_names = []

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
        mic_input = st.audio_input("Micro", key="audio_input")
        if mic_input:
            audio_data.append(mic_input.read())
            file_names.append("Microphone")
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

    # Initialiser l'index de clé si ce n'est pas déjà fait
    if "key_index" not in st.session_state:
        st.session_state["key_index"] = 0

    # Bouton pour lancer la transcription
    if st.button("Transcrire") and audio_data:
        try:
            st.info("Traitement en cours...")
            transcriptions = []
            total_cost = 0
            api_keys = [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]
            key_count = len(api_keys)
            if key_count == 0:
                st.error("Aucune clé API disponible. Veuillez vérifier vos secrets.")
                st.stop()

            for idx, data in enumerate(audio_data):
                seg = AudioSegment.from_file(io.BytesIO(data))
                orig_sec = len(seg)/1000.0
                st.write(f"**Fichier {idx+1} : {file_names[idx]}** - Durée Originale : {human_time(orig_sec)}")

                # Déterminer si la segmentation est nécessaire
                needs_segmentation = False
                if accessibility and model_selection == "Whisper Large" and len(data) > 20*1024*1024:
                    needs_segmentation = True
                elif not accessibility and model_selection == "Whisper Large" and len(data) > 20*1024*1024:
                    needs_segmentation = True

                if needs_segmentation:
                    st.info("Fichier trop volumineux, segmentation en morceaux de 20MB.")
                    # Approximation : durée en secondes pour 20MB
                    # Cela dépend du bitrate, ici on suppose 1 MB ~ 1 seconde
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
                    if st.session_state["key_index"] + 1 >= key_count:
                        st.session_state["key_index"] = 0  # Reset si dépasse
                    key1 = api_keys[st.session_state["key_index"]]
                    key2 = api_keys[(st.session_state["key_index"] + 1) % key_count]
                    st.session_state["key_index"] = (st.session_state["key_index"] + 2) % key_count
                else:
                    key = api_keys[st.session_state["key_index"]]
                    st.session_state["key_index"] = (st.session_state["key_index"] + 1) % key_count

                # Transcription des chunks
                transcripts = []
                costs = []
                for chunk_file in chunk_files:
                    dur_s = len(AudioSegment.from_file(chunk_file))/1000.0
                    if accessibility:
                        # Transcription avec Nova 2
                        trans1 = nova_api.transcribe_audio(chunk_file, key1, "fr", "nova-2")  # Nova 2 ne supporte pas le choix de langue
                        # Vérifier si Nova 2 a réussi
                        if not trans1:
                            st.warning("Transcription Nova 2 échouée, segmentation et retranscription.")
                            # Segmentation et retranscription Nova 2
                            chunk_duration_nova = 20  # secondes
                            nova_chunks = [seg[i*chunk_duration_nova*1000:(i+1)*chunk_duration_nova*1000] for i in range((len(seg) // (chunk_duration_nova*1000)) + 1)]
                            for j, nova_chunk in enumerate(nova_chunks):
                                temp_nova = f"temp_nova_{idx}_{j}.wav"
                                nova_chunk.export(temp_nova, format="wav")
                                trans_retry = nova_api.transcribe_audio(temp_nova, key1, "fr", "nova-2")
                                if trans_retry:
                                    trans1 += " " + trans_retry
                                os.remove(temp_nova)

                        # Transcription avec Whisper Large
                        trans2 = nova_api.transcribe_audio(chunk_file, key2, language_selection, "whisper-large")
                        transcripts.append({"Nova 2": trans1, "Whisper Large": trans2})
                        cost1 = dur_s * 0.007
                        cost2 = dur_s * 0.007
                        costs.extend([cost1, cost2])
                    else:
                        # Transcription unique
                        trans = nova_api.transcribe_audio(chunk_file, key, language_selection if model_selection == "Whisper Large" else "fr", "whisper-large" if model_selection == "Whisper Large" else "nova-2")
                        transcripts.append({model_selection: trans})
                        cost = dur_s * 0.007
                        costs.append(cost)
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
                        st.success("Double transcription terminée.")
                        cL, cR = st.columns(2)
                        with cL:
                            st.subheader("Résultat Nova 2")
                            st.text_area("Texte Nova 2", trans["Nova 2"], height=180)
                            copy_to_clipboard(trans["Nova 2"])
                        with cR:
                            st.subheader("Résultat Whisper Large")
                            st.text_area("Texte Whisper Large", trans["Whisper Large"], height=180)
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
                        st.success("Transcription terminée.")
                        st.text_area("Texte Transcrit", trans[model_selection], height=180)
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

if __name__=="__main__":
    main()
