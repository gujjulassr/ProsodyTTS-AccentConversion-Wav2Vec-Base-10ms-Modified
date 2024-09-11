import os,sys
import librosa
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.to(device)
model.config.output_hidden_states = True
model.eval()


def process_audio_file(audio_file_path, num_zeroes=160):
    y, sr = librosa.load(audio_file_path, sr=16000)
    input_values = processor(y.squeeze(), sampling_rate=sr, return_tensors="pt").input_values
    input_values = input_values.to(device)
    
    zeroes = torch.zeros((1, num_zeroes), dtype=input_values.dtype, device=device)
    input1 = torch.cat((zeroes, input_values), dim=1)
    input2 = torch.cat((input_values, zeroes), dim=1)
    
    return input1, input2, y, sr

def extract_representations(inputs):
    with torch.no_grad():
        outputs = model.wav2vec2(inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
    return last_hidden_state



def pad_to_match(tensor, target_length):
    current_length = tensor.size(1)
    if current_length < target_length:
        padding = (0, target_length - current_length)
        tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
    return tensor



audio_file_path = f'/raid/ai23mtech02001/LJSpeech-1.1_old/wavs/LJ001-0007.wav'
base_name = os.path.basename(audio_file_path).strip().split('.')[0]

input1, input2, y, sr = process_audio_file(audio_file_path, 160)

rep1 = extract_representations(input1)
rep2 = extract_representations(input2)



if rep1.ndim >= 2 and rep1.shape[2] != 768:
    print(f"Filename: {row[0]} - Second dimension is {rep1.shape[1]}, not 768")
    sys.exit()

        # Check for rep2
if rep2.ndim >= 2 and rep2.shape[2] != 768:
    print(f"Filename: {row[0]} - Second dimension is {rep2.shape[1]}, not 768")
    sys.exit()



stacked = torch.stack((rep1.squeeze(0), rep2.squeeze(0)), dim=1)
wav2vec_result = stacked.view(-1, 768).to('cpu')


torch.save(wav2vec_result,f'input_fea_pre/input_feature-{base_name}.pt')

