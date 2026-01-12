import edge_tts
import asyncio
import os

# Map language codes to Edge-TTS voices
VOICE_MAPPING = {
    "vi": {"female": "vi-VN-HoaiMyNeural", "male": "vi-VN-NamMinhNeural"},
    "en": {"female": "en-US-JennyNeural", "male": "en-US-GuyNeural"},
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"}, # Simplified Chinese
    "ja": {"female": "ja-JP-NanamiNeural", "male": "ja-JP-KeitaNeural"},
    "ko": {"female": "ko-KR-SunHiNeural", "male": "ko-KR-InJoonNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
    "de": {"female": "de-DE-KatjaNeural", "male": "de-DE-ConradNeural"},
    "ru": {"female": "ru-RU-SvetlanaNeural", "male": "ru-RU-DmitryNeural"},
    "th": {"female": "th-TH-PremwadeeNeural", "male": "th-TH-NiwatNeural"},
    "id": {"female": "id-ID-GadisNeural", "male": "id-ID-ArdiNeural"},
}

async def generate_speech(text, output_file, voice, rate="+0%", pitch="+0Hz"):
    """
    Generates speech using Edge TTS.
    """
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def get_voice(lang, gender="female"):
    """
    Returns a voice string based on language code.
    Defaults to English if not found.
    """
    # Handle zh-CN / zh-TW if needed, but simple mapping for now
    if lang in VOICE_MAPPING:
        return VOICE_MAPPING[lang].get(gender, VOICE_MAPPING[lang]["female"])

    # Fallback for regional codes (e.g. en-US -> en)
    base_lang = lang.split('-')[0]
    if base_lang in VOICE_MAPPING:
         return VOICE_MAPPING[base_lang].get(gender, VOICE_MAPPING[base_lang]["female"])

    # Default fallback
    return "en-US-JennyNeural"

def generate_audio_segments(segments, output_dir, style="normal", target_lang="vi"):
    """
    Generates audio files for each segment.
    style: 'normal' or 'tvb'
    target_lang: 'vi', 'en', etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []

    # Select Voice based on language
    # For TVB style, we might prefer male or female depending on context,
    # but we stick to female default or make it random/configurable later.
    voice = get_voice(target_lang, "female")

    # TVB Style settings (approximate)
    if style == "tvb":
        rate = "+15%"
        pitch = "+5Hz"
        # If Vietnamese and TVB, we might want to ensure we use a specific expressive voice if available
        # But HoaiMy is good enough.
    else:
        rate = "+0%"
        pitch = "+0Hz"

    print(f"Generating audio (Style: {style}, Lang: {target_lang}, Voice: {voice})...")

    async def _process_all():
        tasks = []
        for i, seg in enumerate(segments):
            text = seg['text']
            # Basic validation
            if not text.strip():
                audio_files.append(None)
                continue

            filename = os.path.join(output_dir, f"seg_{i:04d}.mp3")
            await generate_speech(text, filename, voice, rate, pitch)
            audio_files.append(filename)
        return audio_files

    return asyncio.run(_process_all())
