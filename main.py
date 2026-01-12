import argparse
import os
import shutil
from src.transcriber import transcribe_video
from src.translator import translate_segments
from src.tts_engine import generate_audio_segments
from src.mixer import mix_audio

def main():
    parser = argparse.ArgumentParser(description="AutoDubber - Tool lòng tiếng/thuyết minh tự động")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="output.mp4", help="Path to output video file")
    parser.add_argument("--source_lang", default="auto", help="Source language code (e.g., en, zh)")
    parser.add_argument("--target_lang", default="vi", help="Target language code (default: vi)")
    parser.add_argument("--style", default="normal", choices=["normal", "tvb"], help="Voice style (normal, tvb)")
    parser.add_argument("--mode", default="voiceover", choices=["voiceover", "dubbing"], help="Mode: voiceover (giữ âm gốc nhỏ), dubbing (bỏ âm gốc)")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    # Create temp directory
    temp_dir = "temp_dubbing"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    try:
        # 1. Transcribe
        print("--- Step 1: Transcription ---")
        segments = transcribe_video(args.input, language=args.source_lang)

        # 2. Translate
        print("--- Step 2: Translation ---")
        translated_segments = translate_segments(segments, args.source_lang, args.target_lang)

        # Save transcript for debug
        with open(os.path.join(temp_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            for seg in translated_segments:
                f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")

        # 3. Generate Speech
        print("--- Step 3: Voice Generation ---")
        # Pass target_lang to TTS engine
        audio_files = generate_audio_segments(translated_segments, os.path.join(temp_dir, "audio"), style=args.style, target_lang=args.target_lang)

        # 4. Mix
        print("--- Step 4: Mixing ---")
        mix_audio(args.input, translated_segments, audio_files, args.output, mode=args.mode)

        print("\nSuccess! Video saved to:", args.output)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not args.keep_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
