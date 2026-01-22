from transformers import pipeline
import torch
import numpy as np
import soundfile as sf

# Load TTS pipeline
tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

# Input text
text = "Hello, this is Hugging Face TTS working correctly."

# Generate audio
speech = tts(text)

# Convert to numpy float32
if isinstance(speech["audio"], torch.Tensor):
    audio_array = speech["audio"].cpu().numpy()
else:
    audio_array = np.array(speech["audio"], dtype=np.float32)

# Ensure 1D (mono channel)
if audio_array.ndim > 1:
    audio_array = audio_array.squeeze()

# Save as WAV
sf.write("output.wav", audio_array, speech["sampling_rate"], subtype="PCM_16")

print("✅ Audio saved as output.wav")
