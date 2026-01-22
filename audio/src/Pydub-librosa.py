import librosa
import numpy as np
import queue
import sounddevice as sd
import soundfile as sf

# File processing part
audio_path = "C:\\Emphasis\\FirstProject\\audio\\src\\sample-3s.mp3"
audio_data, sample_rate = librosa.load(audio_path, sr=None)
print(f"Audio loaded: {len(audio_data)} samples at {sample_rate} Hz")

# Export to WAV
sf.write("C:\\Emphasis\\FirstProject\\audio\\src\\sample-3s.wav", audio_data, sample_rate)

from faster_whisper import WhisperModel

# Transcribe the audio file
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("C:\\Emphasis\\FirstProject\\audio\\src\\sample-3s.wav", beam_size=1, vad_filter=True)
print("Detected language:", info.language)
print("Transcription:")
for segment in segments:
    print(segment.text)
    print(segment.start, segment.end)

def setup_realtime_audio():
    try:
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            print(f"  {i}: {dev['name']} (inputs: {dev['max_input_channels']}, outputs: {dev['max_output_channels']})")
        
        print(f"\nDefault input device: {sd.default.device[0]}")
        
        SAMPLE_RATE = 16000
        BLOCK_SIZE = 16000*3
        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            audio_queue.put(indata.copy())

        def process_audio():
            buffer = np.array([])
            while True:
                data = audio_queue.get()
                buffer = np.concatenate((buffer, data))
                if len(buffer) >= BLOCK_SIZE:
                    chunk = buffer[:BLOCK_SIZE]
                    buffer = buffer[BLOCK_SIZE:]
                    audio_np = chunk.flatten()
                    try:
                        segments, _ = model.transcribe(audio_np, beam_size=1, vad_filter=True)
                        for segment in segments:
                            handle_segment(segment.text)
                    except Exception as e:
                        print(f"Transcription error: {e}")

        def handle_segment(text):
            print(">>", text)

        print("\nStarting real-time audio processing...")
        with sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=1):
            print("Listening... Press Ctrl+C to stop")
            process_audio()
            
    except Exception as e:
        print(f"Audio device error: {e}")
        print("Real-time audio processing is not available. File processing completed successfully.")
            
    
if __name__ == "__main__":
    setup_realtime_audio()