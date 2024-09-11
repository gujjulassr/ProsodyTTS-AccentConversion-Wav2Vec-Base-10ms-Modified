import tensorflow as tf
import torch
import numpy as np
import librosa,sys
import soundfile as sf

# Load the model
model = tf.keras.models.load_model('best_model.keras')

# Load and convert PyTorch tensor to TensorFlow tensor
pytorch_tensor = torch.load('/raid/ai23mtech02001/FastDecTF/input_fea_pre/input_feature-LJ001-0007.pt').detach()
numpy_array = pytorch_tensor.cpu().numpy()
tf_tensor = tf.convert_to_tensor(numpy_array)

# Add batch dimension
tf_tensor = tf.expand_dims(tf_tensor, 0)

# Create len_mask and adjust its dimensions
len_mask = tf.convert_to_tensor(np.ones(tf_tensor.shape[1]))
len_mask = tf.expand_dims(len_mask, 0)
len_mask = tf.expand_dims(len_mask, -1)



# Predict using the model
preds = model.predict((tf_tensor, len_mask))

# preds should be a Mel spectrogram, so remove the batch dimension if necessary
mel_spectrogram = preds[0]  # Assuming preds shape is (batch_size, time_steps, n_mels)

# Parameters for Mel Spectrogram and Griffin-Lim
n_fft = 1024
n_mels = 80
hop_length = 221
win_length = 551
sample_rate = 22050

# Generate the mel filterbank
mel_basis = librosa.filters.mel(
    sr=sample_rate, 
    n_fft=n_fft, 
    n_mels=n_mels, 
    fmin=0.0, 
    fmax=sample_rate / 2
)

# Invert the Mel spectrogram back to a linear spectrogram (STFT)
# Transpose the Mel spectrogram to align with the mel_basis
mel_spectrogram_T = mel_spectrogram.T  # Shape becomes (80, 838)

torch.save(mel_spectrogram,'pred_mel.pt')

# Now multiply the pseudo-inverse of the mel_basis with the transposed Mel spectrogram
stft = np.dot(np.linalg.pinv(mel_basis), mel_spectrogram_T)  # Shape will be (513, 838)

# Apply the Griffin-Lim algorithm to get the waveform
waveform = librosa.griffinlim(
    stft, 
    n_iter=60, 
    hop_length=hop_length, 
    win_length=win_length
)

# Save the reconstructed audio to a file
sf.write('reconstructed_audio.wav', waveform, sample_rate)

print(f"Reconstructed audio saved as 'reconstructed_audio.wav'")