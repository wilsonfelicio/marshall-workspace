# Bonus: Adding Audio to Your Briefing

## The Pitch

> "What if you didn't have to read this? What if you could listen to your macro briefing while making coffee or on the drive in?"

## How It Works

Three additions to make your briefing speak:

### 1. The TTS Script (already built)

`generate_audio_simple.py` — takes text in, gives MP3 out. 20 lines of Python.

```bash
echo "Gold rallied 1.2% overnight..." | python3 generate_audio_simple.py
# → outputs briefing_audio.mp3
```

### 2. Update the Cron Payload

Add this to the end of your briefing cron payload:

```
After writing the briefing, also run:
python3 /path/to/generate_audio_simple.py "YOUR_BRIEFING_TEXT"

Then send the audio file to the chat.
```

Or keep it simpler — just generate audio separately and send both text + audio.

### 3. The API Call (What's Actually Happening)

It's one HTTP POST to ElevenLabs:

```bash
curl -X POST "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB" \
  -H "xi-api-key: YOUR_ELEVENLABS_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your briefing text here...",
    "model_id": "eleven_turbo_v2_5",
    "voice_settings": {
      "stability": 0.6,
      "similarity_boost": 0.8
    }
  }' \
  --output briefing.mp3
```

That's it. Text goes in, professional-sounding MP3 comes out.

## Cost

- ElevenLabs free tier: 10,000 characters/month (~3-4 briefings)
- Paid plan: $5/month for 30,000 chars (~15-20 briefings)
- A 200-word briefing is ~1,200 characters

## Voice Options

- **Adam** (pNInz6obpgDQGcFmaJgB) — deep, professional, Bloomberg anchor vibe
- **Rachel** — warm, clear, great for summaries
- **Antoni** — casual, conversational

You can clone your own voice too, but that's a rabbit hole for another day.

## Demo Tip

If you have a pre-generated audio file, **play it for the audience** through the room speakers. The reaction is always "wait, an AI wrote AND read that?" It lands every time.
