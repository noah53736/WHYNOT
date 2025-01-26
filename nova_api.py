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

def segment_audio(audio: AudioSegment, segment_length_ms=10*60*1000):
    """
    Divise l'audio en segments de longueur spécifiée (en millisecondes).
    Par défaut, chaque segment est de 10 minutes.
    """
    segments = []
    for i in range(0, len(audio), segment_length_ms):
        segment = audio[i:i+segment_length_ms]
        segments.append(segment)
    return segments

def transcribe_segment(
    segment: AudioSegment,
    key: str,
    language: str,
    model_name: str,
    punctuate: bool,
    numerals: bool
) -> str:
    """
    Transcrit un segment audio en utilisant une clé API spécifique.
    """
    temp_in = "temp_nova_in.wav"
    transcription = ""
    cost = 0.0

    try:
        # Conversion en 16kHz mono WAV
        segment_16k = segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        segment_16k.export(temp_in, format="wav")

        # Calcul du coût pour le segment
        audio_duration_sec = len(segment) / 1000.0
        audio_duration_min = audio_duration_sec / 60.0

        if model_name == "nova-2":
            cost_per_min = 0.0043
        elif model_name == "whisper-large":
            cost_per_min = 0.0048
        else:
            cost_per_min = 0.0043  # Valeur par défaut

        estimated_cost = audio_duration_min * cost_per_min

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
            cost = round(estimated_cost, 4)
        else:
            st.warning(f"Erreur avec la clé {key}: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Exception durant la transcription: {e}")
        traceback.print_exc()
    finally:
        # Nettoyage du fichier temporaire
        if os.path.exists(temp_in):
            os.remove(temp_in)

    return transcription, cost

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
    Retourne la transcription complète, la clé utilisée, et le coût total.
    """
    temp_in = "temp_nova_in.wav"
    transcription_complete = ""
    used_key = ""
    total_cost = 0.0

    try:
        # Conversion en 16kHz mono WAV
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        segments = segment_audio(audio, segment_length_ms=10*60*1000)  # 10 minutes par segment

        for key in api_keys:
            if key not in credits or credits[key] <= 0:
                continue  # Sauter les clés sans crédit

            for segment in segments:
                # Calcul du coût pour le segment
                audio_duration_sec = len(segment) / 1000.0
                audio_duration_min = audio_duration_sec / 60.0

                if model_name == "nova-2":
                    cost_per_min = 0.0043
                elif model_name == "whisper-large":
                    cost_per_min = 0.0048
                else:
                    cost_per_min = 0.0043  # Valeur par défaut

                estimated_cost = audio_duration_min * cost_per_min

                if credits[key] < estimated_cost:
                    st.warning(f"Crédit insuffisant pour la clé {key} pour ce segment.")
                    continue  # Passer à la clé suivante

                # Transcription du segment
                transcription, cost = transcribe_segment(
                    segment=segment,
                    key=key,
                    language=language,
                    model_name=model_name,
                    punctuate=punctuate,
                    numerals=numerals
                )

                if transcription:
                    transcription_complete += transcription + " "
                    total_cost += cost
                    credits[key] -= cost
                else:
                    st.warning(f"Échec de la transcription avec la clé {key} pour ce segment.")
                    break  # Passer à la clé suivante

            if transcription_complete:
                used_key = key
                save_credits(credits)
                break  # Transcription réussie avec cette clé

        if not transcription_complete:
            st.error("Toutes les clés API ont échoué ou ont des crédits insuffisants.")

    except Exception as e:
        st.error(f"Exception durant la transcription: {e}")
        traceback.print_exc()
    finally:
        # Nettoyage du fichier temporaire
        if os.path.exists(temp_in):
            os.remove(temp_in)

    return transcription_complete.strip(), used_key, round(total_cost, 4)
