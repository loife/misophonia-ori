import argparse
import librosa
import torch

import config
from models.fpasst_wrapper import FPaSSTWrapper, PredictionsWrapper
from data.encode import ManyHotEncoder
from data.decode import batched_decode_preds


def sound_event_detection(args):
    """
    Running Sound Event Detection on an audio clip.
    """
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    
    model = PredictionsWrapper(FPaSSTWrapper(), ckpt_path=args.model_path,
                                    seq_model_type=None,
                                    n_classes_strong=len(config.labels))

    model.eval()
    model.to(device)

    sample_rate = 16_000
    segment_duration = 10 
    segment_samples = segment_duration * sample_rate

    # load audio
    (waveform, _) = librosa.core.load(args.audio_file, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)
    waveform_len = waveform.shape[1]

    audio_len = waveform_len / sample_rate  # in seconds
    print("Audio length (seconds): ", audio_len)

    encoder = ManyHotEncoder(config.labels, audio_len=audio_len)

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
        for i in range(0, len(chunks), args.batch_size):
            batch = torch.cat(chunks[i:i + args.batch_size], dim=0)

            mel = model.mel_forward(batch)
            y_strong, _ = model(mel)

            all_predictions.append(y_strong)

    # Concatenate batches
    y_strong = torch.cat(all_predictions, dim=0)

    # Concatenate chunks back into one timeline
    y_strong = torch.cat(torch.unbind(y_strong, dim=0), dim=-1)

    # Convert to probabilities
    y_strong = torch.sigmoid(y_strong)
    y_strong = y_strong.unsqueeze(0)

    (
        scores_unprocessed,
        scores_postprocessed,
        decoded_predictions
    ) = batched_decode_preds(
        y_strong.float(),
        [args.audio_file],
        encoder,
        median_filter=args.median_window,
        thresholds=args.detection_thresholds,
    )

    for th in decoded_predictions:
        print("***************************************")
        print(f"Detected events using threshold {th}:")
        print(decoded_predictions[th].sort_values(by="onset"))
        print("***************************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./checkpoints/onset_fms=0.634.ckpt")
    parser.add_argument('--audio_file', type=str, default="./test_audio/audio.wav")
    parser.add_argument('--detection_thresholds', type=float, default=(0.1, 0.2, 0.5))
    parser.add_argument('--median_window', type=float, default=9)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    sound_event_detection(args)
