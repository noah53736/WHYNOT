import streamlit as st
import os
import json
import random
import time
from datetime import datetime
from pydub import AudioSegment, silence

import nova_api  # Assure-toi que ce module est correctement configuré

# --- Configuration de la Page ---
st.set_page_config(page_title="NBL Audio", layout="wide")

HISTORY_FILE = "historique.json"
CREDITS_FILE = "credits.json"

# Paramètres de transformations
MIN_SILENCE_LEN = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
COST_PER_SEC = 0.007

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

def get_api_keys():
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

def select_api_key(api_keys, used_keys):
    """
    Sélectionne une clé API non utilisée.
    """
    available_keys = [key for key in api_keys if key[0] not in used_keys]
    if not available_keys:
        st.error("Toutes les clés API sont utilisées ou épuisées.")
        return None
    return random.choice(available_keys)

def main():
    init_history()
    if "history" not in st.session_state:
        st.session_state["history"] = load_history()
    history = st.session_state["history"]

    st.title("NBL Audio : Transcription")
    st.write("Choisissez votre source audio (fichier, micro, ou multi), configurez les options dans la barre latérale, puis lancez la transcription.")

    # Barre latérale : options
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
    lang_choice = st.sidebar.radio("Langue:", ["fr", "en", "es", "de", "it", "pt", "ja", "ko", "zh"])

    # Récupération des clés API depuis les secrets
    api_keys = get_api_keys()
    if not api_keys:
        st.sidebar.error("Aucune clé API disponible. Veuillez ajouter des clés dans les Secrets de l'application.")
        st.stop()

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
                file_name = st.text_input("Nom du Fichier (Optionnel)", upf.name)
                segments.append((audio_data, file_name if file_name else upf.name))
    elif input_type == "Micro (Enregistrement)":
        mic_input = st.audio_input("Micro")
        if mic_input:
            audio_data = mic_input.read()
            st.audio(audio_data, format=mic_input.type)
            file_name = st.text_input("Nom (Optionnel)", "micro.wav")
            segments.append((audio_data, file_name if file_name else "micro.wav"))
    elif input_type == "Multi-Fichiers":
        st.write("Chargez plusieurs fichiers d'un coup (max 5):")
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
        st.write("Enregistrez plusieurs micros (max 5):")
        for i in range(1, 6):
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
            if double_mode:
                # Besoin de deux clés différentes
                available_keys = [key for key in api_keys if key[0] not in used_keys]
                if len(available_keys) < 2:
                    st.error("Pas assez de clés API disponibles pour la double transcription.")
                    os.remove(local_path)
                    os.remove(transformed_path)
                    continue
                selected_keys = random.sample(available_keys, 2)
                key1, name1 = selected_keys[0]
                key2, name2 = selected_keys[1]
                used_keys.extend([key1, key2])
            else:
                # Besoin d'une seule clé
                available_keys = [key for key in api_keys if key[0] not in used_keys]
                if not available_keys:
                    st.error("Pas assez de clés API disponibles pour la transcription.")
                    os.remove(local_path)
                    os.remove(transformed_path)
                    continue
                selected_key = random.choice(available_keys)
                key1, name1 = selected_key
                used_keys.append(key1)

            # Transcription
            try:
                if double_mode:
                    # Double transcription: Nova2 + WhisperLarge
                    st.info("Transcription Nova2 en cours...")
                    transcript_nova2, success_nova2 = nova_api.transcribe_audio(
                        transformed_path,
                        key1,
                        language=lang_choice,
                        model_name="nova-2"
                    )
                    if not success_nova2:
                        st.warning("Erreur avec Nova2. Passage à une autre clé si disponible.")
                        # Sélectionner une autre clé
                        remaining_keys = [key for key in api_keys if key[0] not in used_keys]
                        if remaining_keys:
                            key1_new, name1_new = remaining_keys[0]
                            transcript_nova2, success_nova2 = nova_api.transcribe_audio(
                                transformed_path,
                                key1_new,
                                language=lang_choice,
                                model_name="nova-2"
                            )
                            if success_nova2:
                                used_keys.append(key1_new)
                        else:
                            st.error("Toutes les clés Nova2 sont épuisées.")
                            transcript_nova2 = "Erreur de transcription Nova2."

                    if success_nova2:
                        st.success("Transcription Nova2 terminée.")
                    else:
                        st.error("Transcription Nova2 échouée.")

                    st.info("Transcription WhisperLarge en cours...")
                    transcript_whisper, success_whisper = nova_api.transcribe_audio(
                        transformed_path,
                        key2,
                        language=lang_choice,
                        model_name="whisper-large"
                    )
                    if not success_whisper:
                        st.warning("Erreur avec WhisperLarge. Passage à une autre clé si disponible.")
                        # Sélectionner une autre clé
                        remaining_keys = [key for key in api_keys if key[0] not in used_keys]
                        if remaining_keys:
                            key2_new, name2_new = remaining_keys[0]
                            transcript_whisper, success_whisper = nova_api.transcribe_audio(
                                transformed_path,
                                key2_new,
                                language=lang_choice,
                                model_name="whisper-large"
                            )
                            if success_whisper:
                                used_keys.append(key2_new)
                        else:
                            st.error("Toutes les clés WhisperLarge sont épuisées.")
                            transcript_whisper = "Erreur de transcription WhisperLarge."

                    if success_whisper:
                        st.success("Transcription WhisperLarge terminée.")
                    else:
                        st.error("Transcription WhisperLarge échouée.")

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
                        "Alias/Nom": rename,
                        "Méthode": "Nova 2",
                        "Modèle": "nova-2",
                        "Durée": human_time(orig_sec),
                        "Temps": human_time(time.time() - start_time),
                        "Coût": f"${final_sec * COST_PER_SEC * 2:.2f}",
                        "Transcription": {
                            "Nova2": transcript_nova2,
                            "WhisperLarge": transcript_whisper
                        },
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    # Simple transcription
                    st.info(f"Transcription {model_choice} en cours...")
                    transcript, success = nova_api.transcribe_audio(
                        transformed_path,
                        key1,
                        language=lang_choice,
                        model_name=model_map[model_choice]
                    )
                    if not success:
                        st.warning(f"Erreur avec {model_choice}. Passage à une autre clé si disponible.")
                        # Sélectionner une autre clé
                        remaining_keys = [key for key in api_keys if key[0] not in used_keys]
                        if remaining_keys:
                            key1_new, name1_new = remaining_keys[0]
                            transcript, success = nova_api.transcribe_audio(
                                transformed_path,
                                key1_new,
                                language=lang_choice,
                                model_name=model_map[model_choice]
                            )
                            if success:
                                used_keys.append(key1_new)
                        else:
                            st.error(f"Échec de la transcription {model_choice} avec toutes les clés disponibles.")
                            transcript = f"Erreur de transcription {model_choice}."

                    if success:
                        st.success("Transcription terminée.")
                        st.text_area("", transcript, height=150)
                        copy_to_clipboard(transcript)

                        # Sauvegarder dans l'historique
                        transcriptions.append({
                            "Alias/Nom": rename,
                            "Méthode": model_choice,
                            "Modèle": model_map[model_choice],
                            "Durée": human_time(orig_sec),
                            "Temps": human_time(time.time() - start_time),
                            "Coût": f"${final_sec * COST_PER_SEC:.2f}",
                            "Transcription": transcript,
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
            except Exception as e:
                st.error(f"Erreur lors de la transcription pour le segment #{idx+1}: {e}")
            finally:
                # Nettoyage des fichiers temporaires
                if os.path.exists(local_path):
                    os.remove(local_path)
                if os.path.exists(transformed_path):
                    os.remove(transformed_path)

        # Sauvegarder toutes les transcriptions
        save_history(transcriptions)
        elapsed_time = time.time() - start_time
        st.success(f"Toutes les transcriptions sont terminées en {elapsed_time:.1f}s.")

        # Rafraîchir la page pour mettre à jour l'historique
        st.experimental_rerun()

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

    if __name__ == "__main__":
        main()
    ```

#### **b. Module `nova_api.py`**

Assure-toi que ce module utilise les clés API de manière sécurisée et gère les erreurs pour permettre la rotation des clés en cas d'échec.

```python
import os
import requests
import streamlit as st
import traceback
from pydub import AudioSegment

def transcribe_audio(
    file_path: str,
    dg_api_key: str,
    language: str = "fr",
    model_name: str = "nova-2",
    punctuate: bool = True,
    numerals: bool = True
) -> (str, bool):
    """
    Envoie le fichier audio à DeepGram pour transcription (Nova ou Whisper).
    Retourne la transcription et un booléen indiquant le succès.
    """
    temp_in = "temp_audio.wav"
    try:
        # Convertir en 16kHz WAV
        audio = AudioSegment.from_file(file_path)
        audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_16k.export(temp_in, format="wav")

        # Paramètres
        params = {
            "language": language,
            "model": model_name,
            "punctuate": "true" if punctuate else "false",
            "numerals": "true" if numerals else "false"
        }
        qs = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"https://api.deepgram.com/v1/listen?{qs}"

        headers = {
            "Authorization": f"Token {dg_api_key}",
            "Content-Type": "audio/wav"
        }
        with open(temp_in, "rb") as f:
            payload = f.read()

        resp = requests.post(url, headers=headers, data=payload)
        if resp.status_code == 200:
            j = resp.json()
            return (
                j.get("results", {})
                 .get("channels", [{}])[0]
                 .get("alternatives", [{}])[0]
                 .get("transcript", ""),
                True
            )
        else:
            st.error(f"[DeepGram] Erreur {resp.status_code} : {resp.text}")
            if resp.status_code == 401:
                # Invalid credentials
                return "", False
            return "", False
    except Exception as e:
        st.error(f"[DeepGram] Exception : {e}")
        traceback.print_exc()
        return "", False
    finally:
        if os.path.exists(temp_in):
            os.remove(temp_in)
