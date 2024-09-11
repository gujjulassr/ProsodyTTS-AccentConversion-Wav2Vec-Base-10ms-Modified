import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the ground truth Mel spectrogram from the .pt file
mel_gt_tensor = torch.load('/raid/ai23mtech02001/FastDecTF/test/mels/LJ001-0007.pt')

# Load the predicted Mel spectrogram from the .pt file
mel_pred_tensor = torch.load('pred_mel.pt').transpose(1,0)  # Adjust the path as needed

# Convert the PyTorch tensors to NumPy arrays
mel_gt_spectrogram = mel_gt_tensor.numpy()
mel_pred_spectrogram = mel_pred_tensor

print(mel_pred_spectrogram.shape,"mel shape",mel_gt_spectrogram.shape)

# Convert the Mel spectrograms to dB if they are not already in dB
mel_gt_spectrogram = librosa.power_to_db(mel_gt_spectrogram, ref=np.max)
mel_pred_spectrogram = librosa.power_to_db(mel_pred_spectrogram, ref=np.max)

# Plot the ground truth and predicted Mel spectrograms one above the other
plt.figure(figsize=(10, 8))

# Plot the ground truth Mel spectrogram
plt.subplot(2, 1, 1)
librosa.display.specshow(mel_gt_spectrogram, sr=22050, hop_length=221, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Ground Truth Mel Spectrogram')

# Plot the predicted Mel spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(mel_pred_spectrogram, sr=22050, hop_length=221, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Predicted Mel Spectrogram')

plt.tight_layout()

# Save the figure
output_file = 'mel_spectrogram_comparison_vertical.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"Mel spectrogram comparison saved as '{output_file}'")