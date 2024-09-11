import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the Mel spectrogram from the .pt file
mel_tensor = torch.load('test/mels/LJ001-0007.pt')

# Convert the PyTorch tensor to a NumPy array
mel_spectrogram = mel_tensor.numpy()

# If your Mel spectrogram is not in dB, convert it using librosa
mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Plot the Mel spectrogram (assuming it's already in dB)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram, sr=22050, hop_length=221, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# Save the figure
output_file = 'mel_spectrogram.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"Mel spectrogram saved as '{output_file}'")