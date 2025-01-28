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

def select_api_keys(api_keys, used_keys=[]):
    selected = []
    for key in api_keys:
        if key not in used_keys:
            selected.append(key)
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
        st.sidebar.write("### Aperçus Audio")
        for en in reversed(history[-3:]):
            st.sidebar.markdown(f"**{en['Alias/Nom']}** – {en['Date']}")
            ab = bytes.fromhex(en["Audio Binaire"])
            st.sidebar.audio(ab, format="audio/wav")
    else:
        st.sidebar.info("Historique vide.")

def main():
    st.set_page_config(page_title="N-B-L Audio", layout="wide")
    st.title("N-B-L Audio : Transcription Grand Public")

    init_history()
    history = st.session_state["history"]

    st.write("## Options de Transcription")
    col1, col2 = st.columns([1,1])
    with col1:
        model_selection = st.selectbox("Modèle :", ["Nova 2", "Whisper Large"])
    with col2:
        language_selection = st.selectbox("Langue :", ["fr", "en"])

    model_map = {"Nova 2":"nova-2","Whisper Large":"whisper-large"}
    chosen_model = model_map.get(model_selection,"nova-2")

    accessibility = st.checkbox("Double Transcription")

    st.write("---")
    st.write("## Entrée Audio")
    audio_data = None
    file_name = None

    input_type = st.radio("Type d'Entrée :", ["Fichier (Upload)", "Micro (Enregistrement)"])
    if input_type=="Fichier (Upload)":
        upf = st.file_uploader("Importer l'audio", type=["mp3","wav","m4a","ogg","webm"], accept_multiple_files=True)
        if upf:
            if any(file.size > 300*1024*1024 for file in upf):
                st.warning("Un ou plusieurs fichiers dépassent 300MB.")
            else:
                audio_data = [file.read() for file in upf]
                for file in upf:
                    st.audio(file, format=file.type)
                file_name = st.text_input("Nom du(s) Fichier(s) (Optionnel)")
    else:
        mic_input = st.audio_input("Micro")
        if mic_input:
            audio_data = [mic_input.read()]
            st.audio(mic_input, format=mic_input.type)
            file_name = st.text_input("Nom du Fichier (Optionnel)")

    cA, cB = st.columns(2)
    with cA:
        if st.button("Effacer l'Entrée"):
            audio_data = None
            st.experimental_rerun()
    with cB:
        if st.button("Vider l'Historique"):
            st.session_state["history"] = []
            st.sidebar.success("Historique vidé.")

    if audio_data:
        try:
            # Gestion des fichiers multiples et segmentation si nécessaire
            transcriptions = []
            total_cost = 0
            api_keys = [st.secrets[k] for k in st.secrets if k.startswith("NOVA")]
            used_keys = []

            for idx, data in enumerate(audio_data):
                seg = AudioSegment.from_file(io.BytesIO(data))
                orig_sec = len(seg)/1000.0
                st.write(f"**Fichier {idx+1}** - Durée Originale : {human_time(orig_sec)}")

                # Segmentation intelligente si le fichier > 20MB (~20MB en WAV est approximativement 20MB, ajuster si nécessaire)
                if len(data) > 20*1024*1024:
                    st.info("Fichier trop volumineux, segmentation en morceaux de 20MB.")
                    # Note: Une segmentation plus précise peut être nécessaire
                    chunks = [seg[i:i+20*1024*1024] for i in range(0, len(seg), 20*1024*1024)]
                else:
                    chunks = [seg]

                # Exporter les segments en fichiers temporaires
                chunk_files = []
                for i, chunk in enumerate(chunks):
                    temp_filename = f"temp_chunk_{idx}_{i}.wav"
                    chunk.export(temp_filename, format="wav")
                    chunk_files.append(temp_filename)

                # Sélection des clés API
                if accessibility:
                    selected_keys = select_api_keys(api_keys, used_keys=used_keys)
                    if len(selected_keys) < 2:
                        st.error("Pas assez de clés API pour double transcription.")
                        st.stop()
                    used_keys.extend(selected_keys)
                else:
                    selected_keys = select_api_keys(api_keys, used_keys=used_keys)
                    if len(selected_keys) < 1:
                        st.error("Aucune clé API disponible.")
                        st.stop()

                # Transcription des chunks
                transcripts = []
                costs = []
                for chunk_file in chunk_files:
                    dur_s = len(AudioSegment.from_file(chunk_file))/1000.0
                    if accessibility:
                        key1, key2 = selected_keys
                        trans1 = nova_api.transcribe_audio(chunk_file, key1, language_selection, "nova-2")
                        trans2 = nova_api.transcribe_audio(chunk_file, key2, language_selection, "whisper-large")
                        transcripts.append({"Nova 2": trans1, "Whisper Large": trans2})
                        cost1 = dur_s * 0.007
                        cost2 = dur_s * 0.007
                        costs.extend([cost1, cost2])
                    else:
                        key = selected_keys[0]
                        trans = nova_api.transcribe_audio(chunk_file, key, language_selection, chosen_model)
                        transcripts.append({chosen_model: trans})
                        cost = dur_s * 0.007
                        costs.append(cost)
                    os.remove(chunk_file)

                # Unification des transcriptions
                if accessibility:
                    full_trans_nova = " ".join([t["Nova 2"] for t in transcripts])
                    full_trans_whisper = " ".join([t["Whisper Large"] for t in transcripts])
                    transcriptions.append({"Nova 2": full_trans_nova, "Whisper Large": full_trans_whisper})
                else:
                    full_trans = " ".join([list(t.values())[0] for t in transcripts])
                    transcriptions.append({chosen_model: full_trans})

                total_cost += sum(costs)

            # Affichage des résultats
            if accessibility:
                for idx, trans in enumerate(transcriptions):
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

                    alias1 = generate_alias(6) if not file_name else f"{file_name}_Nova2_{idx+1}"
                    alias2 = generate_alias(6) if not file_name else f"{file_name}_Whisper_{idx+1}"
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
                for idx, trans in enumerate(transcriptions):
                    model = chosen_model.capitalize()
                    st.success(f"Transcription du Fichier {idx+1} terminée.")
                    st.text_area(f"Texte Transcrit - Fichier {idx+1}", list(trans.values())[0], height=180)
                    copy_to_clipboard(list(trans.values())[0])

                    alias = generate_alias(6) if not file_name else f"{file_name}_{model}_{idx+1}"
                    entry = {
                        "Alias/Nom": alias,
                        "Méthode": model,
                        "Modèle": chosen_model,
                        "Durée": human_time(orig_sec),
                        "Temps": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Coût": f"${costs[idx]:.2f}",
                        "Transcription": list(trans.values())[0],
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Audio Binaire": data.hex()
                    }
                    st.session_state["history"].append(entry)

            st.success("Historique mis à jour.")

        except Exception as e:
            st.error(f"Erreur lors de la transcription : {e}")

    display_history(st.session_state["history"])

if __name__=="__main__":
    main()
