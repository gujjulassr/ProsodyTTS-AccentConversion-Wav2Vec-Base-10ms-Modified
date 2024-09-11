import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load an audio file as a waveform `y` and the sample rate `sr`
audio_path = 'reconstructed_audio.wav'
y, sr = librosa.load(audio_path, sr=22050)  # Load with a sample rate of 22.05 kHz

# Parameters for Mel spectrogram
n_fft = 1024        # Length of the FFT window
hop_length = 221    # Number of samples between successive frames
win_length = 551    # Length of the window for FFT
n_mels = 80         # Number of Mel bands

# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, 
                                                 win_length=win_length, n_mels=n_mels)


# Convert to dB (log scale)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Plot the Mel spectrogram with an enhanced colormap
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# Save the figure
output_file = 'mel_spectrogram.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"Mel spectrogram saved as '{output_file}'")