# nova_api.py

import os
import requests
import streamlit as st
import traceback
from pydub import AudioSegment
import io
import json

def load_credits():
    if os.path.exists("credits.json"):
        with open("credits.json", "r") as f:
            credits = json.load(f)
    else:
        # Initialiser les crédits si le fichier n'existe pas
        credits = {}
    return credits

def save_credits(credits):
    with open("credits.json", "w") as f:
        json.dump(credits, f, indent=4)

def transcribe_nova_one_shot(
    file_bytes: bytes,
    api_keys: list,  # Liste des clés API dans l'ordre de priorité
    language: str = "fr",
    model_name: str = "nova-2",
    punctuate: bool = True,
    numerals: bool = True,
    credits: dict = None
) -> (str, str, float):
    """
    Tente de transcrire en utilisant les clés API fournies dans l'ordre.
    Retourne la transcription, la clé utilisée, et le coût.
    """
    temp_in = "temp_nova_in.wav"
    transcription = ""
    used_key = ""
    cost = 0.0

    try:
        # Conversion en 16kHz mono WAV
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_16k.export(temp_in, format="wav")

        for key in api_keys:
            if key not in credits or credits[key] <= 0:
                continue  # Sauter les clés sans crédit

            # Calcul du coût pour le modèle
            if model_name == "nova-2":
                cost_per_min = 0.0043
            elif model_name == "whisper-large":
                cost_per_min = 0.0048
            else:
                cost_per_min = 0.0043  # Valeur par défaut

            # Estimer la durée de l'audio en minutes
            audio_duration_sec = len(audio) / 1000.0
            audio_duration_min = audio_duration_sec / 60.0
            estimated_cost = audio_duration_min * cost_per_min

            if credits[key] < estimated_cost:
                continue  # Sauter si crédit insuffisant

            # Préparation des paramètres de la requête
            params = {
                "language": language,
                "model": model_name,
                "punctuate": "true" if punctuate else "false",
                "numerals": "true" if numerals else "false"
            }
            qs = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"https://api.deepgram.com/v1/listen?{qs}"

            headers = {
                "Authorization": f"Token {key}",
                "Content-Type": "audio/wav"
            }

            with open(temp_in, "rb") as f:
                payload = f.read()

            response = requests.post(url, headers=headers, data=payload)

            if response.status_code == 200:
                result = response.json()
                transcription = (
                    result.get("results", {})
                          .get("channels", [{}])[0]
                          .get("alternatives", [{}])[0]
                          .get("transcript", "")
                )
                used_key = key
                cost = round(estimated_cost, 4)
                # Déduire le coût du crédit
                credits[key] -= cost
                save_credits(credits)
                break  # Transcription réussie
            else:
                st.warning(f"Erreur avec la clé {key}: {response.status_code} - {response.text}")
                continue  # Passer à la clé suivante

        if not transcription:
            st.error("Toutes les clés API ont échoué ou ont des crédits insuffisants.")
    except Exception as e:
        st.error(f"Exception durant la transcription: {e}")
        traceback.print_exc()
    finally:
        # Nettoyage du fichier temporaire
        if os.path.exists(temp_in):
            os.remove(temp_in)

    return transcription, used_key, cost
