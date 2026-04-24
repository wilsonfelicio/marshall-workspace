#!/usr/bin/env python3
"""
Simple ElevenLabs TTS Generator
================================
Converts text to speech using ElevenLabs API.
Outputs an MP3 file ready to send via Telegram.

Usage:
    python3 generate_audio_simple.py "Your text here"
    python3 generate_audio_simple.py < briefing.txt

Requires: ElevenLabs API key (set below or via ELEVENLABS_API_KEY env var)
No pip install needed — pure stdlib.
"""

import urllib.request
import json
import sys
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_KEY")

# Voice: Adam — deep, professional, Bloomberg-anchor vibe
VOICE_ID = "pNInz6obpgDQGcFmaJgB"

# Model: Turbo v2.5 — fast, good quality
MODEL_ID = "eleven_turbo_v2_5"

# Output file
OUTPUT_FILE = "briefing_audio.mp3"

# ---------------------------------------------------------------------------
# GENERATE
# ---------------------------------------------------------------------------

def generate_audio(text: str, output_path: str = OUTPUT_FILE):
    """Send text to ElevenLabs, save MP3."""

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    payload = json.dumps({
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.8,
            "style": 0.3,
            "use_speaker_boost": True
        }
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST", headers={
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    })

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            audio_data = resp.read()

        with open(output_path, "wb") as f:
            f.write(audio_data)

        size_kb = len(audio_data) / 1024
        print(f"✅ Audio saved: {output_path} ({size_kb:.1f} KB)")
        return output_path

    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"❌ API error {e.code}: {error_body}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Get text from argument or stdin
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("Usage: python3 generate_audio_simple.py \"Your text here\"")
        print("   or: echo \"text\" | python3 generate_audio_simple.py")
        sys.exit(1)

    if API_KEY == "YOUR_ELEVENLABS_KEY":
        print("⚠️  Set ELEVENLABS_API_KEY or edit the script with your key")
        sys.exit(1)

    print(f"🎙️  Generating audio ({len(text)} chars)...")
    generate_audio(text)
