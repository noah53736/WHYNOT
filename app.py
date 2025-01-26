import streamlit as st
import os
import io
import random
import string
import time
import subprocess

from datetime import datetime
from pydub import AudioSegment, silence
import requests

# ============================
# Constants & Setup
# ============================
DG_MODELS = [
    "nova-2",
    "whisper-base",
    "whisper-medium",
    "whisper-large",
]

DG_COST_PER_MINUTE = 0.007  # e.g. $0.007/min for Nova. Adjust or remove if you don't want cost tracking.

st.set_page_config(page_title="Deepgram Audio Transcriber", layout="wide")


# ============================
# Helper Functions
# ============================
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
    """Accelerate or slow down audio using ffmpeg & pydub."""
    if abs(factor - 1.0) < 1e-2:
        return audio_seg
    tmp_in  = "temp_in.wav"
    tmp_out = "temp_out.wav"
    audio_seg.export(tmp_in, format="wav")

    remain= factor
    filters = []
    # If user picks >2.0 or <0.5, chain multiple 'atempo' filters
    while remain > 2.0:
        filters.append("atempo=2.0")
        remain /= 2.0
    while remain < 0.5:
        filters.append("atempo=0.5")
        remain /= 0.5
    filters.append(f"atempo={remain}")
    f_str = ",".join(filters)

    cmd = ["ffmpeg", "-y", "-i", tmp_in, "-filter:a", f_str, tmp_out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new_seg = AudioSegment.from_file(tmp_out, format="wav")

    # Clean up
    try:
        os.remove(tmp_in)
        os.remove(tmp_out)
    except:
        pass
    return new_seg

def remove_silences_smooth(audio_seg: AudioSegment) -> AudioSegment:
    """Remove long silences with a crossfade for smoothness."""
    MIN_SIL_MS = 700
    SIL_THRESH_DB= -35
    KEEP_SIL_MS=50
    CROSSFADE_MS=50

    segs = silence.split_on_silence(
        audio_seg,
        min_silence_len=MIN_SIL_MS,
        silence_thresh=SIL_THRESH_DB,
        keep_silence=KEEP_SIL_MS
    )
    if not segs:
        return audio_seg
    combined = segs[0]
    for s in segs[1:]:
        combined = combined.append(s, crossfade=CROSSFADE_MS)
    return combined

def transcribe_deepgram(file_bytes: bytes,
                        dg_api_key: str,
                        model_name: str = "nova-2",
                        language: str = "fr",
                        punctuate: bool = True,
                        numerals: bool = True) -> str:
    """
    Send audio data to Deepgram. 
    Model can be 'nova-2' or 'whisper-large' etc.
    Returns the transcript (str).
    """
    tmp_in = "temp_dg_in.wav"

    # Convert user bytes to 16k mono PCM for best results
    seg = AudioSegment.from_file(io.BytesIO(file_bytes))
    seg_16k= seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    seg_16k.export(tmp_in, format="wav")

    params = []
    params.append(f"model={model_name}")
    if language.strip():
        params.append(f"language={language}")
    if punctuate:
        params.append("punctuate=true")
    if numerals:
        params.append("numerals=true")

    qs= "?"+ "&".join(params)
    url= "https://api.deepgram.com/v1/listen" + qs
    with open(tmp_in,"rb") as f:
        payload= f.read()

    heads={
        "Authorization": f"Token {dg_api_key}",
        "Content-Type": "audio/wav"
    }
    resp= requests.post(url, headers=heads, data=payload)
    if resp.status_code==200:
        j= resp.json()
        alt= j.get("results",{}).get("channels",[{}])[0].get("alternatives",[{}])[0]
        return alt.get("transcript","")
    else:
        st.error(f"[Deepgram] HTTP {resp.status_code}: {resp.text}")
        return ""

# ============================
# MAIN APP
# ============================
def main():
    st.title("Deepgram Transcription App")

    # 1) Acquire your Deepgram key from environment var 'NOVA'
    dg_key = os.getenv("NOVA","")
    if not dg_key:
        st.error("No environment variable 'NOVA' found. Please set it to your Deepgram key.")
        st.stop()

    # 2) Let user pick model
    st.sidebar.header("Deepgram Model Options")
    chosen_model = st.sidebar.selectbox("Model name", DG_MODELS, index=0)
    language = st.sidebar.text_input("Language code", "fr")
    do_punct = st.sidebar.checkbox("Punctuate?", True)
    do_nums  = st.sidebar.checkbox("Numerals => digits?", True)

    # 3) Audio transformations
    st.sidebar.write("---")
    st.sidebar.header("Transformations")
    remove_sil = st.sidebar.checkbox("Remove silences (smooth)?", False)
    speed_fac = st.sidebar.slider("Acceleration Factor", 0.5,4.0,1.0,0.1)

    # 4) Audio input method
    st.write("## Audio Input")
    input_choice = st.radio("Choose input:", ["Upload file", "Microphone"])
    audio_data = None

    if input_choice=="Upload file":
        upf = st.file_uploader("Upload an audio file", type=["mp3","wav","m4a","ogg","webm"])
        if upf is not None:
            audio_data = upf.read()
            st.audio(upf)
    else:
        mic_in = st.audio_input("Record from mic")
        if mic_in:
            audio_data = mic_in.read()
            st.audio(audio_data)

    # 5) If audio_data present, show transformations & transcribe button
    if audio_data:
        # Display a "Transform & Transcribe" button
        if st.button("Transcribe Now"):
            start_t= time.time()

            # Transform
            seg= AudioSegment.from_file(io.BytesIO(audio_data))
            original_sec= len(seg)/1000.0
            if remove_sil:
                seg= remove_silences_smooth(seg)
            if abs(speed_fac -1.0)>1e-2:
                seg= accelerate_ffmpeg(seg, speed_fac)

            # Convert back to bytes
            import io
            bufp= io.BytesIO()
            seg.export(bufp, format="wav")
            new_bytes= bufp.getvalue()

            # Transcribe with Deepgram
            final_txt= transcribe_deepgram(
                new_bytes,
                dg_api_key= dg_key,
                model_name= chosen_model,
                language= language,
                punctuate= do_punct,
                numerals= do_nums
            )

            final_sec= len(seg)/1000.0
            gain_s=0
            if final_sec<original_sec:
                gain_s= original_sec - final_sec

            # Example cost
            minutes= final_sec/60.0
            cost_val= minutes * DG_COST_PER_MINUTE
            cost_str= f"${cost_val:.3f}"

            total_time= time.time()-start_t
            time_str= human_time(total_time)
            st.success(f"Transcribed in {time_str}, gain ~{human_time(gain_s)}, cost ~{cost_str}")

            st.subheader("Transcription")
            st.write(final_txt)

            # Download button
            st.download_button("Download Transcript",
                data= final_txt.encode("utf-8"),
                file_name="transcript.txt",
                mime="text/plain"
            )
    else:
        st.info("Please provide an audio file or record from your mic to transcribe.")


def main_wrapper():
    main()

if __name__ == "__main__":
    main_wrapper()
