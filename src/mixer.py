from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, CompositeVideoClip, AudioClip
import os
import imageio_ffmpeg

def mix_audio(video_path, segments, audio_files, output_path, mode="voiceover"):
    """
    Mixes generated audio with original video using MoviePy.
    mode: 'voiceover' (ducking) or 'dubbing' (remove original)
    """
    print(f"Mixing audio (Mode: {mode})...")

    video = VideoFileClip(video_path)
    original_audio = video.audio

    audio_clips = []

    # 1. Handle Original Audio
    if mode == "voiceover":
        # Lower volume
        if original_audio:
            original_audio = original_audio.with_volume_scaled(0.2) # -14dB approx
            audio_clips.append(original_audio)
    else:
        # Dubbing: discard original audio
        pass

    # 2. Add generated segments
    if segments and audio_files:
        for i, seg in enumerate(segments):
            audio_file = audio_files[i]
            if not audio_file or not os.path.exists(audio_file):
                continue

            # Load speech
            speech_clip = AudioFileClip(audio_file)

            # Set start time
            speech_clip = speech_clip.with_start(seg['start'])

            audio_clips.append(speech_clip)

    # 3. Composite Audio
    print("Compositing audio clips...")

    if not audio_clips:
        # If no audio at all, create a silent audio track
        print("No audio clips to mix. Generating silent audio.")
        # Create silent audio using a lambda
        # duration must be > 0
        dur = video.duration if video.duration > 0 else 1
        final_audio = AudioClip(lambda t: [0, 0], duration=dur, fps=44100)
    else:
        final_audio = CompositeAudioClip(audio_clips)

    # Trim to video duration
    final_audio = final_audio.with_duration(video.duration)

    # 4. Set to video
    final_video = video.with_audio(final_audio)

    # 5. Write output
    print(f"Writing video to {output_path}...")
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print("Done!")
