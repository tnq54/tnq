from deep_translator import GoogleTranslator

def translate_text(text, source_lang='auto', target_lang='vi'):
    """
    Translates text to target language.
    """
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def translate_segments(segments, source_lang='auto', target_lang='vi'):
    """
    Translates a list of segments.
    """
    translated_segments = []
    print(f"Translating {len(segments)} segments to {target_lang}...")

    for seg in segments:
        translated_text = translate_text(seg['text'], source_lang, target_lang)
        new_seg = seg.copy()
        new_seg['text'] = translated_text
        translated_segments.append(new_seg)

    return translated_segments
