# app.py

import warnings
warnings.filterwarnings("ignore", message="invalid escape sequence", category=SyntaxWarning)

import streamlit as st
import os
import requests
import time
import threading
from datetime import datetime
from pydub import AudioSegment, silence
import subprocess

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# -----------------------------------------------------------
# CONFIGURATION ET CONSTANTES
# -----------------------------------------------------------

st.set_page_config(page_title="Double Transcription", layout="wide")

MAX_FILES = 15         # Nombre max de fichiers upload
MAX_SIZE_MB = 200      # Taille max (MB) par fichier
COST_PER_SEC = 0.007   # Coût hypothétique par seconde
SILENCE_LEN_MS = 700   # Paramètres pour remove_silences_classic
SILENCE_THRESH_DB = -35
KEEP_SIL_MS = 50

# Historique en mémoire (pas de fichier credits.json)
if "history" not in st.session_state:
    st.session_state["history"] = []

# -----------------------------------------------------------
# FONCTIONS UTILES
# -----------------------------------------------------------

def human_time(sec: float) -> str:
    """Formatte un nombre de secondes en 'Xs', 'YmZs' ou 'XhYmZs'."""
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

def copy_to_clipboard(text):
    """Petit bouton HTML/JS pour copier un texte donné."""
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

def transcribe_deepgram(file_path: str,
                        api_key: str,
                        language: str,
                        model_name: str) -> (str, bool):
    """
    Envoie le fichier audio à DeepGram pour transcription (Nova II ou Whisper Large).
    Retourne (transcript, success).
    """
    temp_in = "temp_audio_deepgram.wav"
    try:
        # Convertir en 16kHz WAV mono
        audio = AudioSegment.from_file(file_path)
        audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_16k.export(temp_in, format="wav")

        # Paramètres de la requête
        params = {
            "language": language,
            "model": model_name,
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
        st.write(f"[DEBUG] Clé API partielle : {api_key[:4]}****")
        st.write(f"[DEBUG] Modèle : {model_name}")

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

# -----------------------------------------------------------
# CLASSE POUR L'ENREGISTREMENT VIA MICRO
# -----------------------------------------------------------

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorded = AudioSegment.empty()

    def recv(self, frame):
        audio = frame.to_ndarray().flatten()
        self.recorded += AudioSegment(
            data=audio.tobytes(),
            sample_width=2,
            frame_rate=44100,
            channels=1
        )
        return frame

# -----------------------------------------------------------
# APPLICATION STREAMLIT (UN SEUL SCRIPT)
# -----------------------------------------------------------

def main():
    st.title("Application Minimaliste : Nova II + Whisper Large")
    st.write("""
    Téléchargez un ou plusieurs fichiers audio (jusqu'à 15) ou enregistrez via le micro.  
    Pour **chaque** fichier ou enregistrement, on utilise **une clé API** (si disponible)  
    et on fait **2 transcriptions** : Nova II et Whisper Large.
    """)

    # Affichage de l'historique
    with st.sidebar:
        st.header("Historique")
        hist = st.session_state["history"]
        if not hist:
            st.info("Aucune transcription enregistrée.")
        else:
            st.write(f"Total: {len(hist)} transcriptions.")
            for item in reversed(hist[:20]):
                st.markdown(f"- **{item['audio_name']}** / {item['date']}")

    st.sidebar.write("---")
    st.sidebar.header("Options")

    # Langue (pour Whisper Large)
    lang_choice = st.sidebar.radio(
        "Langue pour Whisper Large :",
        ["Français", "Anglais", "Espagnol", "Allemand", "Italien", "Portugais", "Japonais", "Coréen", "Chinois"],
        index=0
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

    remove_sil = st.sidebar.checkbox("Supprimer les silences", value=False)
    speed_factor = st.sidebar.slider("Accélération Audio", 0.5, 4.0, 1.0, 0.1)

    # Clés API DeepGram
    st.sidebar.header("Clés API DeepGram")
    # On charge toutes les clés OVA1..OVA15 dans st.secrets
    api_keys = []
    for i in range(1,16):
        k_name = f"OVA{i}"
        if k_name in st.secrets:
            api_keys.append(st.secrets[k_name])
    if not api_keys:
        st.sidebar.error("Pas de clés API (OVA1..OVA15) trouvées dans secrets.")
        st.stop()
    st.sidebar.write(f"{len(api_keys)} clés trouvées.")

    # Option d'enregistrement via micro
    st.write("## Enregistrement via le Microphone")
    if st.button("Enregistrer via le Microphone"):
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )
        if webrtc_ctx.audio_processor:
            audio_seg = webrtc_ctx.audio_processor.recorded
            if len(audio_seg) > 0:
                # Sauvegarde locale
                in_path = "recorded_audio.wav"
                audio_seg.export(in_path, format="wav")
                segments = [("Microphone Enregistré", in_path)]
                st.audio(in_path, format="audio/wav")
            else:
                st.warning("Aucun enregistrement détecté.")
        else:
            st.warning("Enregistrement non disponible.")

    # Upload
    st.write("## Fichiers Audio")
    upf = st.file_uploader("Importer jusqu'à 15 fichiers audio :",
                           type=["mp3","wav","m4a","ogg","webm"],
                           accept_multiple_files=True)
    segments = []
    if upf:
        if len(upf) > MAX_FILES:
            st.warning(f"Plus de {MAX_FILES} fichiers. On ne garde que les {MAX_FILES} premiers.")
        for fobj in upf[:MAX_FILES]:
            if fobj.size > MAX_SIZE_MB * 1024 * 1024:
                st.warning(f"{fobj.name} dépasse {MAX_SIZE_MB} MB, ignoré.")
                continue
            data = fobj.read()
            segments.append((fobj.name, data))
            st.audio(data, format=fobj.type)

    if segments and st.button("Transcrire Maintenant"):
        start_t = time.time()
        # On ne peut transcrire que len(api_keys) fichiers
        if len(segments) > len(api_keys):
            st.warning(f"Vous avez {len(segments)} fichiers et {len(api_keys)} clés API disponibles.\n"
                       "Seuls les premiers fichiers auront une clé (un fichier par clé).")
        seg_to_process = segments[:len(api_keys)]

        for i, (name, audio_bytes) in enumerate(seg_to_process):
            st.write(f"### Fichier #{i+1}: {name}")
            # Renommage de fichier
            new_name = st.text_input(f"Renommer {name} :", value=name, key=f"rename_{i}")
            if not new_name:
                new_name = name
            # Sauvegarde locale
            in_path = f"temp_in_{i}.wav"
            with open(in_path, "wb") as w:
                w.write(audio_bytes)
            try:
                seg = AudioSegment.from_file(in_path)
                orig_dur = len(seg)/1000.0
                if remove_sil:
                    seg = remove_silences_classic(seg)
                if abs(speed_factor - 1.0) > 1e-2:
                    seg = accelerate_ffmpeg(seg, speed_factor)
                final_dur = len(seg)/1000.0
                st.write(f"**Durée finale :** {human_time(final_dur)}")
                out_path = f"temp_out_{i}.wav"
                seg.export(out_path, format="wav")
                # Affichage de l'audio transformé
                transformed_data = open(out_path, "rb").read()
                st.audio(transformed_data, format="audio/wav")
            except Exception as e:
                st.error(f"Erreur audio : {e}")
                if os.path.exists(in_path):
                    os.remove(in_path)
                continue

            # Récupère la clé API pour ce fichier
            key_api = api_keys[i]

            # Placeholders pour transcriptions côte à côte
            cols = st.columns(2)
            place_nova = cols[0].empty()
            place_whisper = cols[1].empty()

            # Fonctions
            def run_nova(path, api_key, placeholder):
                txt, ok = transcribe_deepgram(path, api_key, "fr", "nova-2")
                if ok:
                    placeholder.success("Nova II terminée")
                    placeholder.text_area("Nova II", txt, height=150)
                    copy_to_clipboard(txt)
                    # Ajout dans hist
                    st.session_state["history"].append({
                        "audio_name": new_name,
                        "double_mode": True,
                        "model_nova2": "nova-2",
                        "transcript_nova2": txt,
                        "model_whisper": "whisper-large",
                        "transcript_whisper": "",
                        "duration": human_time(orig_dur),
                        "elapsed_time": human_time(time.time()-start_t),
                        "cost_estimate": f"${final_dur*COST_PER_SEC*2:.2f}",
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    placeholder.error("Échec transcription Nova II")

            def run_whisper(path, api_key, lang, placeholder):
                txt, ok = transcribe_deepgram(path, api_key, lang, "whisper-large")
                if ok:
                    placeholder.success("Whisper Large terminée")
                    placeholder.text_area("Whisper Large", txt, height=150)
                    copy_to_clipboard(txt)
                    # Mettre à jour hist
                    for item in reversed(st.session_state["history"]):
                        if item["audio_name"] == new_name and item["double_mode"] and not item["transcript_whisper"]:
                            item["transcript_whisper"] = txt
                            break
                else:
                    placeholder.error("Échec transcription Whisper Large")

            # Lancer les transcriptions dans des threads
            t_nova = threading.Thread(target=run_nova, args=(out_path, key_api, place_nova))
            t_whisper = threading.Thread(target=run_whisper, args=(out_path, key_api, selected_lang, place_whisper))
            t_nova.start()
            t_whisper.start()
            t_nova.join()
            t_whisper.join()

            # Nettoyage local
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)

        # Fin de toutes les transcriptions
        elapsed = time.time() - start_t
        st.success(f"Transcriptions terminées en {human_time(elapsed)}.")

if __name__ == "__main__":
    main()
