import librosa
from IPython.display import Audio, display
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os

# Audio preprocessing
N_FFT = 400
HOP_LENGTH=160
N_MELS=80
SAMPLE_RATE=16000

# Whisper parameters
N_SAMPLES = 30 * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 10ms per audio frame
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 20ms per audio token

#
# Dataset Paths
#

def get_datasets_path():
    return os.path.dirname(os.path.dirname(__file__)) # Outside of the repo

def get_dataset_path(name):
    return os.path.join(get_datasets_path(), name)

#
# Audio Utils
#
    
def load_audio(pathOrTensor):
    if isinstance(pathOrTensor, str):
        y, _ = librosa.load(pathOrTensor, sr=SAMPLE_RATE, mono=True) # I have found that torchaudio sometimes can't open some wav files
        return torch.from_numpy(y)
    else:
        return pathOrTensor

mel_filters = torch.from_numpy(np.load(os.path.join(os.path.dirname(__file__), "assets",  "mel_filters.npz"), allow_pickle=False)["mel_80"])

def spectogram(audio):

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=False)
    magnitudes = torch.sum((stft ** 2), dim=-1)[..., :-1]

    # Mel
    mel_spec = mel_filters.to(audio.device) @ magnitudes

    # Log
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec

def reverbrate(waveforms, rir, preprocess = False):
    assert len(waveforms.shape) == 1 # Only single dimension is allowed
    assert len(rir.shape) == 1 # Only single dimension is allowed

    # Find start of a RIR and cut the begining
    if preprocess:
        # Find peak index
        _, direct_index = rir.abs().max(axis=0, keepdim=True)

        # Cut from after peak
        rir = rir[direct_index:]

    # Source length
    source_len = waveforms.shape[0]

    # Normalize remaining
    rir = rir / torch.norm(rir, p=2) # Mean square

    # NOTE: THIS ALL NOT NEEDED for fftconvolve
    # Flip for convolution (we are adding previous values (aka "echo") for each point
    # rir = torch.flip(rir,[0])

    # Pad with zeros to match output time
    # waveforms = torch.cat((torch.zeros(rir.shape[0]-1,dtype=waveforms.dtype), waveforms), 0)

    # Calculate convolution
    waveforms = waveforms.unsqueeze(0).unsqueeze(0)
    rir = rir.unsqueeze(0).unsqueeze(0)
    # waveforms = torch.nn.functional.conv1d(waveforms, rir)
    waveforms = torchaudio.functional.fftconvolve(waveforms, rir)
    waveforms = waveforms.squeeze(dim=0).squeeze(dim=0)
    waveforms = waveforms[0:source_len]

    return waveforms

def smooth_sound_detector(detections, max_duration):
    assert len(detections.shape) == 1 # Only single dimension is allowed
    output = torch.zeros(detections.shape[0])
    counter = 0
    for i in range(0, detections.shape[0]):
        output[i] = detections[i]
        if detections[i] == 0:
            counter = counter + 1
        else:
            if (counter > 0) and (counter <= max_duration):
                for j in range(i-counter, i):
                    output[j] = 1                    
            counter = 0
    return output

def sound_detector(waveform, frame_size, treshold):
    waveform = waveform.unfold(-1, frame_size, frame_size)
    return (waveform.abs().max(dim=-1).values > treshold).float()

#
# Tensor utils
#

def sliding_window(tensor, window_size, step):

    # Load last dimension
    last_dim = tensor.size(-1)
    if window_size > last_dim:
         raise ValueError("Window size is larger than the tensor's last dimension")

    # Create sliding window
    unfolded = tensor.unfold(-1, window_size, step)

    # Permute dimensions
    total_dims = tensor.dim()
    dims = []
    dims.append(total_dims-1)
    for i in range(total_dims - 1):
        dims.append(i)
    dims.append(total_dims)
    unfolded = unfolded.permute(*dims)

    return unfolded

#
# Notebook debugging
#

def audio(fn):
    if isinstance(fn, str):
        return display(Audio(filename=fn, rate=SAMPLE_RATE))
    else:
        waveform = fn.numpy()
        return display(Audio(data=waveform, rate=SAMPLE_RATE))
        
def plot_waveform(waveform, title="Waveform", xlim=None, ylim=None):
    assert len(waveform.shape) == 1
    waveform = waveform.numpy()
    num_frames = waveform.shape[0]
    time_axis = torch.arange(0, num_frames) / SAMPLE_RATE
    figure, axes = plt.subplots(1, 1)
    axes.plot(time_axis, waveform, linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    if xlim:
      axes.set_xlim(xlim)
    if ylim:
      axes.set_ylim(ylim)
    plt.show(block=False)

#
# Math Utils
#

def clamp(value, lower, upper):
    return lower if value < lower else upper if value > upper else value