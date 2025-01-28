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

# --- set_page_config doit être le premier appel Streamlit ---
st.set_page_config(page_title="NBL Audio", layout="wide")

TRANSCRIPTIONS_JSON = "transcriptions.json"

# Paramètres de transformations
MIN_SILENCE_LEN = 700
SIL_THRESH_DB    = -35
KEEP_SIL_MS      = 50
COST_PER_SEC     = 0.007

def init_transcriptions():
    if not os.path.exists(TRANSCRIPTIONS_JSON):
        with open(TRANSCRIPTIONS_JSON,"w",encoding="utf-8") as f:
            json.dump([],f,indent=4)

def load_transcriptions():
    with open(TRANSCRIPTIONS_JSON,"r",encoding="utf-8") as f:
        return json.load(f)

def save_transcriptions(lst):
    with open(TRANSCRIPTIONS_JSON,"w",encoding="utf-8") as f:
        json.dump(lst,f,indent=4)

def generate_alias(length=5):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

def human_time(sec: float)-> str:
    s = int(sec)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        m, s2 = divmod(s, 60)
        return f"{m}m{s2}s"
    else:
        h, r = divmod(s, 3600)
        m, s3 = divmod(r, 60)
        return f"{h}h{m}m{s3}s"

def accelerate_ffmpeg(seg: AudioSegment, factor: float)-> AudioSegment:
    if abs(factor - 1.0) < 1e-2:
        return seg
    tmp_in = "temp_in_acc.wav"
    tmp_out = "temp_out_acc.wav"
    seg.export(tmp_in, format="wav")
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
    cmd = ["ffmpeg","-y","-i",tmp_in,"-filter:a",f_str, tmp_out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new_seg = AudioSegment.from_file(tmp_out, format="wav")
    try:
        os.remove(tmp_in)
        os.remove(tmp_out)
    except:
        pass
    return new_seg

def remove_silences_classic(seg: AudioSegment,
                            min_silence_len=MIN_SILENCE_LEN,
                            silence_thresh=SIL_THRESH_DB,
                            keep_silence=KEEP_SIL_MS):
    parts = silence.split_on_silence(
        seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    if not parts:
        return seg
    comb = parts[0]
    for p in parts[1:]:
        comb = comb.append(p, crossfade=0)
    return comb

def copy_to_clipboard(text):
    # Petit script HTML/JS
    sc = f"""
    <script>
    function copyText(txt){{
        navigator.clipboard.writeText(txt).then(
            ()=>alert('Texte copié!'),
            (err)=>alert('Échec copie: '+err)
        );
    }}
    </script>
    <button onclick="copyText(`{text}`)" style="margin-top:5px;">Copier</button>
    """
    st.components.v1.html(sc)

def display_history_side():
    st.sidebar.header("Historique")
    st.sidebar.write("---")
    tlist = load_transcriptions()
    if not tlist:
        st.sidebar.info("Aucune transcription enregistrée.")
    else:
        st.sidebar.write(f"Total: {len(tlist)} transcriptions.")
        for item in tlist[::-1][:8]:
            st.sidebar.markdown(f"- **{item.get('audio_name','?')}** / {item.get('date','?')}")
            # Optionnel : Ajouter un aperçu audio
            # Assure-toi que le fichier audio est accessible ou stocke le binaire si nécessaire

def get_available_api_keys():
    """
    Récupère toutes les clés API disponibles depuis les secrets.
    Retourne une liste de tuples (clé, indice).
    """
    api_keys = []
    for i in range(1,16):
        key_name = f"NOVA{i}"
        try:
            key = st.secrets[key_name]
            api_keys.append( (key, i) )
        except KeyError:
            st.sidebar.error(f"Clé API manquante : {key_name}")
    return api_keys

def main():
    init_transcriptions()
    display_history_side()

    st.title("NBL Audio : Transcription")
    st.markdown("Choisissez votre source audio (fichier, micro, ou multi), configurez les options dans la barre latérale, puis lancez la transcription.")

    # Barre latérale : transformations, double mode, modèle, langue
    st.sidebar.header("Options")
    remove_sil = st.sidebar.checkbox("Supprimer les silences", False)
    speed_factor = st.sidebar.slider("Accélération Audio", 0.5, 4.0, 1.0, 0.1)
    double_mode = st.sidebar.checkbox("Transcription Double (Nova2 + Whisper)", False)

    st.sidebar.write("---")
    st.sidebar.header("Modèle (si simple)")
    model_choice = st.sidebar.radio("Modèle IA:", ["Nova 2", "Whisper Large"])
    model_map = {"Nova 2": "nova-2", "Whisper Large": "whisper-large"}

    st.sidebar.write("---")
    st.sidebar.header("Langue")
    lang_choice = st.sidebar.radio("Langue:", ["fr", "en"])

    # Récupération des clés API depuis st.secrets
    api_keys = get_available_api_keys()
    if not api_keys:
        st.sidebar.error("Aucune clé API disponible. Veuillez ajouter des clés dans les Secrets de l'application.")
        st.stop()

    # Initialisation de l'état des clés API (clé disponible ou non)
    if 'available_keys' not in st.session_state:
        st.session_state.available_keys = api_keys.copy()

    # Choix du type d'entrée
    st.write("## Mode d'Entrée")
    mode_in = st.radio("", ["Fichier (Upload)", "Micro (Enregistrement)", "Multi-Fichiers", "Multi-Micro"])

    segments = []  # Liste de tuples (AudioSegment, nom)

    if mode_in == "Fichier (Upload)":
        upf = st.file_uploader("Fichier audio (mp3, wav, m4a, ogg, webm)", type=["mp3","wav","m4a","ogg","webm"])
        if upf:
            st.audio(upf, format=upf.type)
            rename = st.text_input("Renommer (optionnel)", upf.name)
            segments.append((upf, rename if rename else upf.name))

    elif mode_in == "Micro (Enregistrement)":
        mic = st.audio_input("Enregistrement Micro")
        if mic:
            st.audio(mic, format=mic.type)
            rename = st.text_input("Nom (optionnel)", "micro.wav")
            segments.append((mic, rename if rename else "micro.wav"))

    elif mode_in == "Multi-Fichiers":
        st.write("Chargez plusieurs fichiers d'un coup (max 5):")
        many_files = st.file_uploader("Importer plusieurs fichiers", accept_multiple_files=True, type=["mp3","wav","m4a","ogg","webm"])
        if many_files:
            for idx, fobj in enumerate(many_files):
                st.audio(fobj, format=fobj.type)
                rename = st.text_input(f"Renommer Fichier #{idx+1} (optionnel)", fobj.name, key=f"rename_f_{idx}")
                segments.append((fobj, rename if rename else fobj.name))

    else:  # Multi-Micro
        st.write("Enregistrez plusieurs micros (max 5):")
        for i in range(1,6):
            micX = st.audio_input(f"Enregistrement Micro #{i}", key=f"mic_{i}")
            if micX:
                st.audio(micX, format=micX.type)
                rename = st.text_input(f"Nom Micro #{i} (optionnel)", f"micro_{i}.wav", key=f"rename_m_{i}")
                segments.append((micX, rename if rename else f"micro_{i}.wav"))

    # Bouton "Transcrire"
    if len(segments) > 0 and st.button("Transcrire Maintenant"):
        transcriptions = load_transcriptions()
        start_time = time.time()
        used_api_indices = []  # Pour éviter de réutiliser la même clé plusieurs fois simultanément

        for idx, (segObj, rename) in enumerate(segments):
            st.write(f"### Segment #{idx+1}: {rename}")
            # Sauvegarder localement
            local_path = f"temp_input_{idx}.wav"
            with open(local_path, "wb") as ff:
                ff.write(segObj.read())
            # Charger AudioSegment
            try:
                seg = AudioSegment.from_file(local_path)
                if remove_sil:
                    seg = remove_silences_classic(seg)
                if abs(speed_factor - 1.0) > 1e-2:
                    seg = accelerate_ffmpeg(seg, speed_factor)
                # Export transformé
                transformed_path = f"temp_transformed_{idx}.wav"
                seg.export(transformed_path, format="wav")
            except Exception as e:
                st.error(f"Erreur lors des transformations pour le segment #{idx+1}: {e}")
                os.remove(local_path)
                continue

            # Sélection des clés API
            if double_mode:
                # Besoin de deux clés différentes
                available_indices = [i for i in range(len(st.session_state.available_keys)) if i not in used_api_indices]
                if len(available_indices) < 2:
                    st.error("Pas assez de clés API disponibles pour la double transcription.")
                    os.remove(local_path)
                    os.remove(transformed_path)
                    continue
                selected_indices = random.sample(available_indices, 2)
                key1, idx1 = st.session_state.available_keys[selected_indices[0]]
                key2, idx2 = st.session_state.available_keys[selected_indices[1]]
                used_api_indices.extend([selected_indices[0], selected_indices[1]])
            else:
                # Besoin d'une seule clé
                available_indices = [i for i in range(len(st.session_state.available_keys)) if i not in used_api_indices]
                if not available_indices:
                    st.error("Pas assez de clés API disponibles pour la transcription.")
                    os.remove(local_path)
                    os.remove(transformed_path)
                    continue
                selected_index = random.choice(available_indices)
                key1, idx1 = st.session_state.available_keys[selected_index]
                used_api_indices.append(selected_index)

            # Transcription
            try:
                if double_mode:
                    # Double transcription: Nova2 + WhisperLarge
                    st.info("Transcription Nova2 en cours...")
                    transcript_nova2, success_nova2 = nova_api.transcribe_audio(transformed_path, key1, language=lang_choice, model_name="nova-2")
                    if not success_nova2:
                        st.warning("Erreur avec Nova2. Passage à la clé suivante.")
                        # Relancer avec une autre clé si disponible
                        remaining_keys = [k for k in st.session_state.available_keys if k not in used_api_indices]
                        if remaining_keys:
                            key1_new, idx1_new = remaining_keys[0]
                            transcript_nova2, success_nova2 = nova_api.transcribe_audio(transformed_path, key1_new, language=lang_choice, model_name="nova-2")
                            if success_nova2:
                                used_api_indices.append(st.session_state.available_keys.index(remaining_keys[0]))
                            else:
                                st.error("Échec de la transcription Nova2 avec toutes les clés disponibles.")
                                transcript_nova2 = "Erreur de transcription Nova2."

                    st.success("Transcription Nova2 terminée.")
                    
                    st.info("Transcription WhisperLarge en cours...")
                    transcript_whisper, success_whisper = nova_api.transcribe_audio(transformed_path, key2, language=lang_choice, model_name="whisper-large")
                    if not success_whisper:
                        st.warning("Erreur avec WhisperLarge. Passage à la clé suivante.")
                        # Relancer avec une autre clé si disponible
                        remaining_keys = [k for k in st.session_state.available_keys if k not in used_api_indices]
                        if remaining_keys:
                            key2_new, idx2_new = remaining_keys[0]
                            transcript_whisper, success_whisper = nova_api.transcribe_audio(transformed_path, key2_new, language=lang_choice, model_name="whisper-large")
                            if success_whisper:
                                used_api_indices.append(st.session_state.available_keys.index(remaining_keys[0]))
                            else:
                                st.error("Échec de la transcription WhisperLarge avec toutes les clés disponibles.")
                                transcript_whisper = "Erreur de transcription WhisperLarge."

                    st.success("Transcription WhisperLarge terminée.")
                    
                    # Affichage des résultats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Nova2")
                        st.text_area("", transcript_nova2, height=150)
                        copy_to_clipboard(transcript_nova2)
                    with col2:
                        st.subheader("WhisperLarge")
                        st.text_area("", transcript_whisper, height=150)
                        copy_to_clipboard(transcript_whisper)
                    
                    # Sauvegarder dans l'historique
                    transcriptions.append({
                        "audio_name": rename,
                        "double_mode": True,
                        "transcript_nova2": transcript_nova2,
                        "transcript_whisper": transcript_whisper,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    # Simple transcription
                    st.info(f"Transcription {model_choice} en cours...")
                    transcript, success = nova_api.transcribe_audio(transformed_path, key1, language=lang_choice, model_name=model_map[model_choice])
                    if not success:
                        st.warning(f"Erreur avec {model_choice}. Passage à la clé suivante.")
                        # Relancer avec une autre clé si disponible
                        remaining_keys = [k for k in st.session_state.available_keys if k not in used_api_indices]
                        if remaining_keys:
                            key1_new, idx1_new = remaining_keys[0]
                            transcript, success = nova_api.transcribe_audio(transformed_path, key1_new, language=lang_choice, model_name=model_map[model_choice])
                            if success:
                                used_api_indices.append(st.session_state.available_keys.index(remaining_keys[0]))
                            else:
                                st.error(f"Échec de la transcription {model_choice} avec toutes les clés disponibles.")
                                transcript = f"Erreur de transcription {model_choice}."
                    
                    if success:
                        st.success("Transcription terminée.")
                    
                    st.text_area("", transcript, height=150)
                    copy_to_clipboard(transcript)
                    
                    # Sauvegarder dans l'historique
                    transcriptions.append({
                        "audio_name": rename,
                        "double_mode": False,
                        "model": model_map[model_choice],
                        "transcript": transcript,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
            except Exception as e:
                st.error(f"Erreur lors de la transcription pour le segment #{idx+1}: {e}")
            finally:
                # Nettoyage des fichiers temporaires
                os.remove(local_path)
                os.remove(transformed_path)

        # Sauvegarder toutes les transcriptions
        save_transcriptions(transcriptions)
        elapsed_time = time.time() - start_time
        st.success(f"Toutes les transcriptions sont terminées en {elapsed_time:.1f}s.")

        # Rafraîchir la page pour mettre à jour l'historique
        st.experimental_rerun()

if __name__ == "__main__":
    main()
