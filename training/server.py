from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import torch
import os
import numpy as np
import librosa
from urllib.parse import unquote

from models.fpasst_wrapper import FPaSSTWrapper, PredictionsWrapper
from data.encode import ManyHotEncoder

import config

app = Flask(__name__)
CORS(app)

sample_rate = 16_000
segment_duration = 10 
segment_samples = segment_duration * sample_rate

model_path="./checkpoints/onset_fms=0.634.ckpt"
median_window=9
batch_size=8

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = PredictionsWrapper(FPaSSTWrapper(), ckpt_path=model_path,
                                seq_model_type=None,
                                n_classes_strong=len(config.labels))
model.eval()
model.to(device)

def np_sigmoid(z):
    return 1/(1 + np.exp(-z))

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

@app.route('/', methods=['POST'])
def index():
    audio_file = unquote(request.json['path'])
    (waveform, _) = librosa.core.load(audio_file, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)
    waveform_len = waveform.shape[1]

    audio_len = waveform_len / sample_rate  # in seconds
    print("Audio length (seconds): ", audio_len)

    num_chunks = waveform_len // segment_samples + (waveform_len % segment_samples != 0)

    chunks = []
    for i in range(num_chunks):
        start_idx = i * segment_samples
        end_idx = min((i + 1) * segment_samples, waveform_len)

        chunk = waveform[:, start_idx:end_idx]

        # Pad final chunk
        if chunk.shape[1] < segment_samples:
            pad_size = segment_samples - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))

        chunks.append(chunk)

    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = torch.cat(chunks[i:i + batch_size], dim=0)

            mel = model.mel_forward(batch)
            y_strong, _ = model(mel)

            all_predictions.append(y_strong)

    # Concatenate batches
    y_strong = torch.cat(all_predictions, dim=0)

    # Concatenate chunks back into one timeline
    y_strong = torch.cat(torch.unbind(y_strong, dim=0), dim=-1)

    # Convert to probabilities
    y_strong = torch.sigmoid(y_strong)
    
    return jsonify({"probs" : y_strong.squeeze().tolist()})

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8000)