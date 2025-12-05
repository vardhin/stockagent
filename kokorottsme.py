from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
import numpy as np

# Initialize pipeline once (reusable)
pipeline = KPipeline(lang_code='a')

def speak(text, voice='af_heart', speed=1, save_file=None):
    """
    Convert text to speech and stream audio.
    
    Args:
        text: Text to convert to speech
        voice: Voice to use (default: 'af_heart')
        speed: Speech speed (default: 1)
        save_file: Optional filename to save audio (e.g., 'output.wav' or 'output.mp3')
    """
    generator = pipeline(text, voice=voice, speed=speed)
    all_audio = []
    
    for i, (gs, ps, audio) in enumerate(generator):
        # Convert tensor to numpy array
        if hasattr(audio, 'numpy'):
            audio_np = audio.numpy()
        else:
            audio_np = np.array(audio)
        
        all_audio.append(audio_np)
        
        # Stream: Play this chunk immediately
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(), 
            frame_rate=24000,
            sample_width=2,
            channels=1
        )
        print(f"Playing chunk {i}...")
        play(audio_segment)
    
    # Save if filename provided
    if save_file:
        complete_audio = np.concatenate(all_audio)
        
        if save_file.endswith('.wav'):
            sf.write(save_file, complete_audio, 24000)
            print(f"Saved {save_file}")
        elif save_file.endswith('.mp3'):
            audio_int16 = (complete_audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_int16.tobytes(), 
                frame_rate=24000,
                sample_width=2,
                channels=1
            )
            audio_segment.export(save_file, format='mp3')
            print(f"Saved {save_file}")

# Example usage
if __name__ == "__main__":
    speak("Hello, this is a test of the Kokoro text to speech system.", 
          save_file='output.mp3')