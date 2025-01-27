import streamlit as st
import os
import io
import time
import random
import string
import json
import subprocess
from datetime import datetime
from pydub import AudioSegment, silence

import nova_api

HISTORY_FILE = "historique.json"
CREDITS_FILE = "credits.json"
MIN_SILENCE_LEN = 700
SIL_THRESH_DB = -35
KEEP_SIL_MS = 50
COST_PER_SEC = 0.007
CHUNK_LIMIT_MB = 180

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
        json.dump(hist,
