# app.py

import streamlit as st
import os
import json
import random
import string
import time
import subprocess
from datetime import datetime
from pydub import AudioSegment, silence

import nova_api

# --- Configuration de la Page ---
st.set_page_config(page_title="NBL Audio", layout="wide")

HISTORY_FILE = "historique.json"

# Paramètres de transformations
MIN_SILENCE_LEN = 700  # en ms
SIL_THRESH_DB = -35     # en dB
KEEP_SIL_MS = 50        # en ms
COST_PER_SEC = 0.007    # Coût par seconde de transcription

def init_history():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f, indent=4)

def load_history():
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
    cmd = ["ffmpeg", "-y", "-i", tmp_in, "-filter:a", f_str, tmp_out]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        new_seg = AudioSegment.from_file(tmp_out, format="wav")
    except Exception as e:
        st.error(f"Erreur lors de l'accélération audio : {e}")
        new_seg = audio_seg
    finally:
        if os.path.exists(tmp_in):
            os.remove(tmp_in)
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
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

def display_history_side():
    st.sidebar.write("---")
    st.sidebar.header("Historique")
    history = load_history()
    if not history:
        st.sidebar.info("Aucune transcription enregistrée.")
    else:
        st.sidebar.write(f"Total: {len(history)} transcriptions.")
        for item in history[::-1][:8]:
            st.sidebar.markdown(f"- **{item.get('audio_name','?')}** / {item.get('date','?')}")

def get_available_api_keys():
    """
    Récupère toutes les clés API disponibles depuis les secrets.
    Retourne une liste de tuples (clé, nom).
    """
    api_keys = []
    for i in range(1, 16):
        key_name = f"NOVA{i}"
        try:
            key = st.secrets[key_name]
            api_keys.append((key, key_name))
        except KeyError:
            st.sidebar.error(f"Clé API manquante : {key_name}")
    return api_keys

def reset_app():
    if st.button("Recommencer à zéro"):
        st.session_state["history"] = []
        save_history([])
        st.experimental_rerun()

def main():
    init_history()
    display_history_side()

    st.title("NBL Audio : Transcription")
    st.markdown("Choisissez votre source audio (fichier, micro, ou multi), configurez les options dans la barre latérale, puis lancez la transcription.")

    st.sidebar.write("---")
    st.sidebar.header("Paramètres de Transcription")

    # Sélection de la langue (applicable uniquement pour Whisper Large)
    lang_choice = st.sidebar.radio(
        "Langue de Whisper Large :",
        options=["Français", "Anglais", "Espagnol", "Allemand", "Italien", "Portugais", "Japonais", "Coréen", "Chinois"],
        horizontal=True
    )
    lang_map = {
        "Français": "fr",
        "Anglais": "en",
        "Espagnol": "es",
        "Allemand": "de",
        "Italien": "it",
        "Portugais": "pt",
        "Japonais": "ja",
        "Coréen": "ko",
        "Chinois": "zh"
    }
    selected_lang = lang_map.get(lang_choice, "fr")

    # Autres options
    remove_sil = st.sidebar.checkbox("Supprimer les silences", False)
    speed_factor = st.sidebar.slider("Accélération Audio", 0.5, 4.0, 1.0, 0.1)

    st.sidebar.write("---")

    # Récupération des clés API depuis les secrets
    api_keys = get_available_api_keys()
    if not api_keys:
        st.sidebar.error("Aucune clé API disponible. Veuillez ajouter des clés dans les Secrets de l'application.")
        st.stop()

    # Double transcription activée par défaut
    double_mode = True  # Toujours activé

    # Choix du mode d'entrée
    st.write("## Mode d'Entrée")
    input_type = st.radio("", ["Fichier (Upload)", "Micro (Enregistrement)", "Multi-Fichiers", "Multi-Micro"])
    segments = []  # Liste de tuples (fichier, nom)

    if input_type == "Fichier (Upload)":
        upf = st.file_uploader("Importer l'audio", type=["mp3","wav","m4a","ogg","webm"])
        if upf:
            if upf.size > 200 * 1024 * 1024:
                st.warning("Fichier > 200MB (limite Streamlit).")
            else:
                audio_data = upf.read()
                st.audio(audio_data, format=upf.type)
                file_name = st.text_input("Nom du Fichier (Optionnel)", upf.name, key="rename_single")
                segments.append((audio_data, file_name if file_name else upf.name))
    elif input_type == "Micro (Enregistrement)":
        mic_input = st.audio_input("Micro")
        if mic_input:
            audio_data = mic_input.read()
            st.audio(audio_data, format=mic_input.type)
            file_name = st.text_input("Nom (Optionnel)", "micro.wav", key="rename_micro_single")
            segments.append((audio_data, file_name if file_name else "micro.wav"))
    elif input_type == "Multi-Fichiers":
        st.write("Chargez plusieurs fichiers d'un coup (max 15):")
        many_files = st.file_uploader("Importer plusieurs fichiers", accept_multiple_files=True, type=["mp3","wav","m4a","ogg","webm"])
        if many_files:
            for idx, fobj in enumerate(many_files):
                if fobj.size > 200 * 1024 * 1024:
                    st.warning(f"Fichier #{idx+1} > 200MB (limite Streamlit).")
                    continue
                audio_data = fobj.read()
                st.audio(audio_data, format=fobj.type)
                rename = st.text_input(f"Renommer Fichier #{idx+1} (optionnel)", fobj.name, key=f"rename_f_{idx}")
                segments.append((audio_data, rename if rename else fobj.name))
    else:  # Multi-Micro
        st.write("Enregistrez plusieurs micros (max 15):")
        for i in range(1, 16):
            micX = st.audio_input(f"Enregistrement Micro #{i}", key=f"mic_{i}")
            if micX:
                audio_data = micX.read()
                st.audio(audio_data, format=micX.type)
                rename = st.text_input(f"Nom Micro #{i} (Optionnel)", f"micro_{i}.wav", key=f"rename_m_{i}")
                segments.append((audio_data, rename if rename else f"micro_{i}.wav"))

    # Bouton "Transcrire"
    if len(segments) > 0 and st.button("Transcrire Maintenant"):
        transcriptions = load_history()
        start_time = time.time()
        used_keys = []  # Pour éviter de réutiliser la même clé simultanément

        for idx, (audio_bytes, rename) in enumerate(segments):
            st.write(f"### Segment #{idx+1}: {rename}")
            # Sauvegarder localement
            local_path = f"temp_input_{idx}.wav"
            with open(local_path, "wb") as ff:
                ff.write(audio_bytes)
            # Charger AudioSegment
            try:
                seg = AudioSegment.from_file(local_path)
                orig_sec = len(seg) / 1000.0
                if remove_sil:
                    seg = remove_silences_classic(seg)
                if abs(speed_factor - 1.0) > 1e-2:
                    seg = accelerate_ffmpeg(seg, speed_factor)
                final_sec = len(seg) / 1000.0
                st.write(f"Durée Finale : {human_time(final_sec)}")
                # Export transformé
                transformed_path = f"temp_transformed_{idx}.wav"
                seg.export(transformed_path, format="wav")
            except Exception as e:
                st.error(f"Erreur lors des transformations pour le segment #{idx+1}: {e}")
                os.remove(local_path)
                continue

            # Sélection des clés API
            available_keys = [key for key in api_keys if key[0] not in used_keys]
            if len(available_keys) < 2:
                st.error("Pas assez de clés API disponibles pour la double transcription.")
                os.remove(local_path)
                os.remove(transformed_path)
                continue
            selected_keys = random.sample(available_keys, 2)
            key_nova2, name_nova2 = selected_keys[0]
            key_whisper, name_whisper = selected_keys[1]
            used_keys.extend([key_nova2, key_whisper])

            # Création des placeholders pour afficher les transcriptions dès qu'elles sont prêtes
            placeholder_nova2 = st.empty()
            placeholder_whisper = st.empty()

            # Transcription Nova II
            def transcribe_nova2():
                transcript, success = nova_api.transcribe_audio(
                    transformed_path,
                    key_nova2,
                    language="fr",  # Langue non nécessaire pour Nova II si elle ne supporte pas la sélection
                    model_name="nova-2"
                )
                if success:
                    placeholder_nova2.success("Transcription Nova II terminée.")
                    placeholder_nova2.text_area("Nova II", transcript, height=150)
                    copy_to_clipboard(transcript)
                    # Ajouter à l'historique
                    transcriptions.append({
                        "audio_name": rename,
                        "double_mode": True,
                        "model_nova2": "Nova II",
                        "transcript_nova2": transcript,
                        "model_whisper": "Whisper Large",
                        "transcript_whisper": "",
                        "duration": human_time(orig_sec),
                        "elapsed_time": human_time(time.time() - start_time),
                        "cost": f"${final_sec * COST_PER_SEC * 2:.2f}",
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    placeholder_nova2.error("Erreur lors de la transcription Nova II.")

            # Transcription Whisper Large
            def transcribe_whisper():
                transcript, success = nova_api.transcribe_audio(
                    transformed_path,
                    key_whisper,
                    language=selected_lang,
                    model_name="whisper-large"
                )
                if success:
                    placeholder_whisper.success("Transcription Whisper Large terminée.")
                    placeholder_whisper.text_area("Whisper Large", transcript, height=150)
                    copy_to_clipboard(transcript)
                    # Mettre à jour l'historique avec la transcription Whisper Large
                    for transcription in transcriptions:
                        if transcription["audio_name"] == rename and transcription["double_mode"]:
                            transcription["transcript_whisper"] = transcript
                            break
                else:
                    placeholder_whisper.error("Erreur lors de la transcription Whisper Large.")

            # Lancer les transcriptions dans des threads pour les exécuter simultanément
            import threading
            thread_nova2 = threading.Thread(target=transcribe_nova2)
            thread_whisper = threading.Thread(target=transcribe_whisper)
            thread_nova2.start()
            thread_whisper.start()

            # Nettoyage des fichiers temporaires après les transcriptions
            thread_nova2.join()
            thread_whisper.join()
            if os.path.exists(local_path):
                os.remove(local_path)
            if os.path.exists(transformed_path):
                os.remove(transformed_path)

        # Sauvegarder toutes les transcriptions
        save_history(transcriptions)
        elapsed_time = time.time() - start_time
        st.success(f"Toutes les transcriptions sont terminées en {human_time(elapsed_time)}.")

if __name__ == "__main__":
    main()
    reset_app()
