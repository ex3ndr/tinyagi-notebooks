import random
from .utils import SAMPLE_RATE, load_audio, reverbrate, sound_detector, smooth_sound_detector
import torch
import torchaudio.functional as F

def select_random_segment(source, length):
    output = torch.zeros(length)

    # If source is equal to the target
    to_offset = 0
    source_offset = 0
    l = length
    if source.shape[0] < length:  # If source is smaller than needed
        to_offset = random.randint(0, length - source.shape[0])
        l = source.shape[0]
    elif source.shape[0] > length: # IF source is bigger than needed
        source_offset = random.randint(0, source.source[0] - length)

    # Apply
    output[to_offset:to_offset+l] = output[to_offset:to_offset+l] + source[source_offset:source_offset+l]

    return output
    

def add_audio_chunk(waveforms, labels, source, speech):
        
    # Calculate offsets
    to_offset = 0
    source_offset = 0
    l = source.shape[0]
    if source.shape[0] < waveforms.shape[0]:
        to_offset = random.randint(0, waveforms.shape[0] - source.shape[0])
        source_offset = 0
        l = source.shape[0]
    elif source.shape[0] > waveforms.shape[0]:
        source_offset = random.randint(0, source.shape[0] - waveforms.shape[0])
        to_offset = 0
        l = waveforms.shape[0]

    # Append
    waveforms[to_offset:to_offset+l] = waveforms[to_offset:to_offset+l] + source[source_offset:source_offset+l]

    # Update labels
    if speech is not None:
        ss = source_offset // 320
        ls = to_offset // 320
        ll = l // 320
        if ll > 0:
            labels[ls : ls + ll] = speech[ss : ss + ll]

def add_audio_noise(waveforms, noise, snr):
    noise = select_random_segment(noise, waveforms.shape[0])
    return F.add_noise(waveforms.unsqueeze(0), noise.unsqueeze(0), torch.tensor([snr]))[0]

def synthesize_sample(duration, 
                      
                      # Background
                      background = None, 
                      background_snr = 10, # Safe default

                      # Voice
                      clean = None, 
                      clean_treshold = 0.01,
                      clean_smooth = 5, # Default for half of the input sequence or ~100ms
                      clean_effector = None,

                      # RIR
                      rir = None):
    waveforms = torch.zeros(SAMPLE_RATE * duration)
    labels = torch.zeros(SAMPLE_RATE * duration // 320)

    # Load background
    if background is not None:
        background = load_audio(background)

    # Add clean sound
    if clean is not None:

        # Load source audio and detect voice
        clean = load_audio(clean)
        detected_voice = sound_detector(clean, 320, clean_treshold)
        detected_voice = smooth_sound_detector(detected_voice, clean_smooth)

        # Reverbrate. This is a environment feature and we
        # apply it before effects that would simulate voice
        # transmission
        if rir is not None:
            rir = load_audio(rir)
            clean = reverbrate(clean, rir)

        # Apply effector
        if clean_effector is not None:
            clean = clean_effector.apply(clean)

        # Add audio chunk
        add_audio_chunk(waveforms, labels, clean, detected_voice)

        # Apply background noise
        if background is not None:
            waveforms = add_audio_noise(waveforms, background, background_snr)
    else:
        # No clean sound: add background as is
        if background is not None:
            add_audio_chunk(waveforms, labels, background, None)

    # Return result
    return waveforms, labels