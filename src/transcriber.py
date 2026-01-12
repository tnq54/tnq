import whisper
import os
import torch
import imageio_ffmpeg

def transcribe_video(video_path, model_size="base", language=None):
    """
    Transcribes video audio using OpenAI Whisper.
    Returns a list of segments: {'start': float, 'end': float, 'text': str}
    """
    print(f"Loading Whisper model ({model_size})...")
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set FFMPEG environment variable for Whisper
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

    model = whisper.load_model(model_size, device=device)

    print(f"Transcribing {video_path}...")

    # Handle 'auto' language
    if language == "auto":
        language = None

    result = model.transcribe(video_path, language=language)

    return result["segments"]
