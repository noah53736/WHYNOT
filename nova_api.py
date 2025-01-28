# nova_api.py

import os
import requests
import streamlit as st
import traceback
from pydub import AudioSegment
import random

class APIKeyManager:
    def __init__(self, api_keys):
        """
        Initialise le gestionnaire de clés API.
        :param api_keys: Liste de tuples (clé, nom)
        """
        self.api_keys = api_keys.copy()
        random.shuffle(self.api_keys)  # Mélange initial des clés pour utilisation aléatoire
        self.failed_keys = set()

    def get_next_key(self):
        """
        Retourne la prochaine clé API disponible.
        :return: Tuple (clé, nom) ou (None, None) si aucune clé valide n'est disponible
        """
        available_keys = [key for key in self.api_keys if key[0] not in self.failed_keys]
        if not available_keys:
            st.error("Toutes les clés API sont épuisées ou invalides.")
            return None, None
        return random.choice(available_keys)

    def mark_key_as_failed(self, key):
        """
        Marque une clé API comme ayant échoué.
        :param key: La clé API à marquer comme échouée
        """
        self.failed_keys.add(key)

def transcribe_audio(
    file_path: str,
    api_key_manager: APIKeyManager,
    language: str = "fr",
    model_name: str = "nova-2",
    punctuate: bool = True,
    numerals: bool = True
) -> (str, bool):
    """
    Envoie le fichier audio à DeepGram pour transcription (Nova ou Whisper).
    Gère la rotation des clés API en cas d'erreur 401.
    :param file_path: Chemin du fichier audio
    :param api_key_manager: Instance de APIKeyManager pour gérer les clés API
    :param language: Langue de la transcription
    :param model_name: Nom du modèle IA à utiliser
    :param punctuate: Booléen pour ajouter la ponctuation
    :param numerals: Booléen pour convertir les chiffres en nombres
    :return: Tuple (transcription, succès)
    """
    temp_in = "temp_audio.wav"
    try:
        # Conversion en 16kHz WAV mono
        audio = AudioSegment.from_file(file_path)
        audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_16k.export(temp_in, format="wav")

        # Paramètres de la requête
        params = {
            "language": language,
            "model": model_name,
            "punctuate": "true" if punctuate else "false",
            "numerals": "true" if numerals else "false"
        }
        qs = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"https://api.deepgram.com/v1/listen?{qs}"

        while True:
            dg_api_key, key_name = api_key_manager.get_next_key()
            if not dg_api_key:
                return "Erreur : Aucune clé API valide disponible.", False

            headers = {
                "Authorization": f"Token {dg_api_key}",
                "Content-Type": "audio/wav"
            }
            with open(temp_in, "rb") as f:
                payload = f.read()

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
            elif resp.status_code == 401:
                st.warning(f"[DeepGram] Clé API invalide ou expirée : {key_name}. Rotation de la clé.")
                api_key_manager.mark_key_as_failed(dg_api_key)
                continue  # Essayer avec une autre clé
            else:
                st.error(f"[DeepGram] Erreur {resp.status_code} : {resp.text}")
                return f"Erreur {resp.status_code} : {resp.text}", False

    except Exception as e:
        st.error(f"[DeepGram] Exception : {e}")
        traceback.print_exc()
        return f"Exception : {e}", False
    finally:
        if os.path.exists(temp_in):
            os.remove(temp_in)
