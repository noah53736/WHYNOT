import streamlit as st
import requests
import os
import time
import threading
from datetime import datetime
from pydub import AudioSegment, silence
import subprocess

# -----------------------------------------------------------
#               CONFIGURATION ET CONSTANTES
# -----------------------------------------------------------

st.set_page_config(page_title="Double Transcription", layout="wide")

# Vous pouvez ajuster au besoin
COST_PER_SEC = 0.007  # Prix hypothétique
MAX_FILES = 15        # Nombre max de fichiers
MAX_SIZE_MB = 200     # Taille max d'un fichier
SILENCE_LEN_MS = 700  # Pour remove_silences_classic
SILENCE_THRESH_DB = -35
KEEP_SIL_MS = 50

# Dans ce script minimaliste, on stocke l'historique dans st.session_state
if "history" not in st.session_state:
    st.session_state["history"] = []

# -----------------------------------------------------------
#         FONCTIONS UTILES (TRANSCRIPTION, ETC.)
# -----------------------------------------------------------

def human_time(sec: float) -> str:
    """Formate un nombre de secondes en Xs, XmYs ou XhYmYs."""
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

def remove_silences_classic(audio_seg: AudioSegment,
                            min_silence_len=SILENCE_LEN_MS,
                            silence_thresh=SILENCE_THRESH_DB,
                            keep_silence=KEEP_SIL_MS):
    """Coupe automatiquement les silences d'un AudioSegment."""
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

def accelerate_ffmpeg(audio_seg: AudioSegment, factor: float) -> AudioSegment:
    """Accélère ou ralentit un AudioSegment via ffmpeg."""
    if abs(factor - 1.0) < 1e-2:
        return audio_seg
    tmp_in = "temp_in_acc.wav"
    tmp_out = "temp_out_acc.wav"
    audio_seg.export(tmp_in, format="wav")
    remain = factor
    filters = []
    # On procède par étapes pour éviter d'aller directement de 1.0 à 4.0
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

def transcribe_deepgram(file_path: str,
                        api_key: str,
                        language: str,
                        model_name: str) -> (str, bool):
    """
    Envoie le fichier audio à DeepGram pour transcription (Nova II ou Whisper Large).
    Retourne (transcription, success).
    """
    temp_in = "temp_audio_deepgram.wav"
    try:
        # Convertir en 16kHz WAV mono
        audio = AudioSegment.from_file(file_path)
        audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_16k.export(temp_in, format="wav")

        # Paramètres de la requête
        params = {
            "language": language,       # ex: "fr"
            "model": model_name,        # "nova-2" ou "whisper-large"
            "punctuate": "true",
            "numerals": "true"
        }
        # Construction de l'URL
        qs = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"https://api.deepgram.com/v1/listen?{qs}"

        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav"
        }

        with open(temp_in, "rb") as f:
            payload = f.read()

        # Debug minimal
        st.write(f"[DEBUG] Clé API utilisée (partielle) : {api_key[:4]}****")
        st.write(f"[DEBUG] Modèle utilisé : {model_name}")

        resp = requests.post(url, headers=headers, data=payload)
        if resp.status_code == 200:
            j = resp.json()
            transcript = (
                j.get("results", {})
                 .get("channels", [{}])[0]
                 .get("alternatives", [{}])[0]
                 .get("transcript", "")
            )
            return transcript, True
        else:
            st.error(f"[DeepGram] Erreur {resp.status_code} : {resp.text}")
            return "", False
    except Exception as e:
        st.error(f"[DeepGram] Exception : {e}")
        return "", False
    finally:
        if os.path.exists(temp_in):
            os.remove(temp_in)

def copy_to_clipboard(text):
    """Petit bouton pour copier la transcription."""
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

# -----------------------------------------------------------
#         INTERFACE PRINCIPALE STREAMLIT
# -----------------------------------------------------------

def main():
    st.title("Application Minimaliste : Double Transcription (Nova II + Whisper Large)")
    st.write("""
    Téléchargez un ou plusieurs fichiers audio (jusqu'à 15).  
    Pour chaque fichier, on utilise **une clé API distincte** (si possible) pour éviter de dépasser les quotas.  
    Puis on fait **2 transcriptions** : Nova II et Whisper Large.
    """)

    # Historique en mémoire
    with st.sidebar:
        st.header("Historique")
        if not st.session_state["history"]:
            st.info("Aucune transcription enregistrée.")
        else:
            st.write(f"Total: {len(st.session_state['history'])} transcriptions.")
            for item in reversed(st.session_state["history"][:15]):
                st.markdown(f"- **{item['audio_name']}** / {item['date']}")

    # Paramètres latéraux
    st.sidebar.write("---")
    st.sidebar.header("Options de Transcription")

    # Langue (pour Whisper Large)
    lang_choice = st.sidebar.radio(
        "Langue (Whisper Large) :",
        ["fr", "en", "es", "de", "it", "pt", "ja", "ko", "zh"],
        index=0
    )
    remove_sil = st.sidebar.checkbox("Supprimer les silences", value=False)
    speed_factor = st.sidebar.slider("Accélération Audio", 0.5, 4.0, 1.0, 0.1)

    # Chargement de clés API
    st.sidebar.write("---")
    st.sidebar.header("Clés API")
    api_keys = []
    for i in range(1, 16):
        key_name = f"OVA{i}"
        try:
            key_val = st.secrets[key_name]
            api_keys.append(key_val)
        except KeyError:
            pass
    if not api_keys:
        st.sidebar.error("Aucune clé trouvée dans st.secrets. Ajoutez OVA1..OVA15.")
        st.stop()

    st.sidebar.write(f"**{len(api_keys)} clés disponibles**")

    # Upload multiple files
    st.write("## Import de fichiers audio")
    upf = st.file_uploader("Sélectionnez jusqu'à 15 fichiers audio :", 
                           type=["mp3","wav","m4a","ogg","webm"], 
                           accept_multiple_files=True)

    # On collecte les segments
    segments = []
    if upf:
        if len(upf) > MAX_FILES:
            st.warning(f"Vous avez importé plus de {MAX_FILES} fichiers. On va ignorer les supplémentaires.")
        for idx, fobj in enumerate(upf[:MAX_FILES]):
            if fobj.size > MAX_SIZE_MB * 1024 * 1024:
                st.warning(f"Le fichier {fobj.name} dépasse {MAX_SIZE_MB} MB. Ignoré.")
                continue
            audio_data = fobj.read()
            st.audio(audio_data, format=fobj.type)
            segments.append((audio_data, fobj.name))

    # Bouton de transcription
    if segments and st.button("Transcrire Maintenant"):
        # On va boucler sur les segments
        start_time = time.time()

        # On limite si on n'a pas assez de clés
        files_to_process = segments[:len(api_keys)]
        if len(segments) > len(api_keys]:
            st.warning(f"Vous avez {len(segments)} fichiers mais seulement {len(api_keys)} clés API disponibles.\n"
                       "Seuls les premiers fichiers auront une clé (un fichier par clé).")

        for i, (audio_bytes, name) in enumerate(files_to_process):
            st.write(f"### Fichier #{i+1} : {name}")
            local_in = f"input_{i}.wav"
            with open(local_in, "wb") as w:
                w.write(audio_bytes)
            # Transformations audio
            try:
                seg = AudioSegment.from_file(local_in)
                orig_dur = len(seg) / 1000.0
                if remove_sil:
                    seg = remove_silences_classic(seg)
                if abs(speed_factor - 1.0) > 1e-2:
                    seg = accelerate_ffmpeg(seg, speed_factor)
                final_dur = len(seg) / 1000.0
                st.write(f"**Durée finale :** {human_time(final_dur)}")
                local_out = f"transformed_{i}.wav"
                seg.export(local_out, format="wav")
            except Exception as e:
                st.error(f"Erreur lecture/transformations : {e}")
                if os.path.exists(local_in):
                    os.remove(local_in)
                continue

            # Récupère la clé pour ce fichier
            key_api = api_keys[i]  
            placeholder_nova = st.empty()
            placeholder_whisper = st.empty()

            # Fonctions de transcription
            def run_nova():
                txt, ok = transcribe_deepgram(local_out, key_api, "fr", "nova-2")
                if ok:
                    placeholder_nova.success("Nova II terminée")
                    placeholder_nova.text_area("Nova II", txt, height=150)
                    copy_to_clipboard(txt)
                    # Historique
                    st.session_state["history"].append({
                        "audio_name": name,
                        "double_mode": True,
                        "model_nova2": "nova-2",
                        "transcript_nova2": txt,
                        "model_whisper": "whisper-large",
                        "transcript_whisper": "",
                        "duration": human_time(orig_dur),
                        "cost_estimate": f"${final_dur * COST_PER_SEC * 2:.2f}",
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    placeholder_nova.error("Échec transcription Nova II")

            def run_whisper():
                txt, ok = transcribe_deepgram(local_out, key_api, lang_choice, "whisper-large")
                if ok:
                    placeholder_whisper.success("Whisper Large terminée")
                    placeholder_whisper.text_area("Whisper Large", txt, height=150)
                    copy_to_clipboard(txt)
                    # Mise à jour historique (on met la transcription whisper sur la même entrée)
                    # On parcourt depuis la fin
                    for item in reversed(st.session_state["history"]):
                        if item.get("audio_name","") == name and item.get("double_mode"):
                            if not item.get("transcript_whisper"):
                                item["transcript_whisper"] = txt
                                break
                else:
                    placeholder_whisper.error("Échec transcription Whisper Large")

            # Lancement multi-threads
            t_nova = threading.Thread(target=run_nova)
            t_whis = threading.Thread(target=run_whisper)
            t_nova.start()
            t_whis.start()
            t_nova.join()
            t_whis.join()

            # Nettoyage
            if os.path.exists(local_in):
                os.remove(local_in)
            if os.path.exists(local_out):
                os.remove(local_out)

        # Fin de toutes les transcriptions
        elapsed = time.time() - start_time
        st.success(f"Transcriptions terminées en {human_time(elapsed)}.")

def run_app():
    main()

if __name__ == "__main__":
    run_app()
