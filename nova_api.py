
# nova_api.py

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
) -> str:
    """
    Transcrit un fichier audio local (file_path) en utilisant DeepGram (dg_api_key).
    Retourne la transcription ou une chaîne vide en cas d'erreur.
    """
    temp_in = "temp_audio.wav"
    try:
        audio = AudioSegment.from_file(file_path)
        audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_16k.export(temp_in, format="wav")

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
                 .get("transcript", "")
            )
        else:
            st.error(f"[DeepGram] Erreur {resp.status_code} : {resp.text}")
            return ""
    except Exception as e:
        st.error(f"[DeepGram] Exception : {e}")
        traceback.print_exc()
        return ""
    finally:
        if os.path.exists(temp_in):
            os.remove(temp_in)
