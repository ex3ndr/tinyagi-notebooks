import numpy as np
import torch
import torchaudio
import json
from torch.utils.data import Dataset
from torch.profiler import record_function
import random
import math
from glob import glob
from .utils import SAMPLE_RATE, TOKENS_PER_SECOND, load_audio
from .synth import synthesize_sample

#
# Common
#

DATASET_SAMPLE_LENGTH = 5

def sample_dataset(dataset):
    return dataset[random.randint(0, len(dataset) - 1)]

#
# Libriparty
#

class LibripartyDataset(Dataset):
    def __init__(self, meta, dir):
        self.data = meta
        self.dir = dir
        self.keys = list(meta.keys())
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):

        # Load session
        intervals = self.data[self.keys[idx]]

        # Source
        with record_function("librosa:load"):
            audio = load_audio(self.dir + "/" + self.keys[idx])

        # Truncate or pad with zeros
        if DATASET_SAMPLE_LENGTH * SAMPLE_RATE < len(audio):
            audio = audio[0:DATASET_SAMPLE_LENGTH * SAMPLE_RATE]
        elif len(audio) < DATASET_SAMPLE_LENGTH * SAMPLE_RATE:
            audio = torch.cat((audio, torch.zeros(DATASET_SAMPLE_LENGTH * SAMPLE_RATE - audio.shape[0])))

        # Target
        target = torch.zeros(DATASET_SAMPLE_LENGTH * TOKENS_PER_SECOND)
        for k in intervals:
            sstart = k[0]
            sstop = k[1]
            sstart = math.floor(sstart * TOKENS_PER_SECOND)
            sstop = math.floor(sstop * TOKENS_PER_SECOND)
            target[sstart:sstop] = 1

        return audio, target
        
def libriparty_dataset(dir):
    return LibripartyDataset(json.load(open(dir + "/meta.json")), dir)

def _resolve(effectOrFunction):
    if isinstance(effectOrFunction, str):
        return effectOrFunction
    elif effectOrFunction is None:
        return None
    else:
        return _resolve(effectOrFunction())

def one_of(*args):
    return lambda:_resolve(random.choice(list(args)))

def maybe(effect, p):
    return lambda: _resolve(effect) if random.random() < p else None

def sequental(*args):
    return lambda: None if (result := ",".join(filter(lambda x: x is not None, map(_resolve, args)))) == "" else result

#
# Synthetic
#

class SyntheticDataset(Dataset):
    def __init__(self, count, speech_dir = None, background_dir = None, rir_dir = None):
        self.count = count

        # Speech
        self.speech_prob = 0.8
        self.speech_files = None
        self.speech_treshold = 0.01
        self.speech_smooth = 5
        if speech_dir is not None:
            self.speech_files = glob(speech_dir + "/*.wav")
        
        # Speech tempo
        self.speech_tempo_min = 0.8
        self.speech_tempo_max = 1.5

        # Background
        self.background_prob = 0.8
        self.background_min_snr = -1
        self.background_max_snr = 30
        self.background_files = None
        if background_dir is not None:
            self.background_files = glob(background_dir + "/*.wav")

        # RIR
        self.rir_prob = 0.8
        self.rir_files = None
        if rir_dir is not None:
            self.rir_files = glob(rir_dir + "/*.wav")

        # Codecs
        self.codecs = [
            {'format': "wav", 'encoder': "pcm_mulaw"},
            {'format': "g722"},
            # {'format': "ogg", 'encoder': "opus", ""}, # Still experimental?
            {"format": "mp3", "codec_config": torchaudio.io.CodecConfig(bit_rate=8_000)},
            {"format": "mp3", "codec_config": torchaudio.io.CodecConfig(bit_rate=64_000)}
        ]
        self.codecs_prob = 0.5

        # Effects
        self.effects = None
        self.effects_prob = 0.8

    def __len__(self):
        return self.count
    def __getitem__(self, idx):
        with record_function("synth_sample"):

            # Parts
            clean = None
            background = None
            background_snr = random.uniform(self.background_min_snr, self.background_max_snr)
            rir = None
            effector = None
            speech_tempo = random.uniform(self.speech_tempo_min, self.speech_tempo_max)
            codec = None
            effect = None
            effector = None

            # Add speech
            if self.speech_files is not None and random.random() < self.speech_prob:
                clean = load_audio(random.choice(self.speech_files))
            
            # Add background
            if self.background_files is not None and (random.random() < self.background_prob or clean is None): # Always pick background if no speech is present
                background = load_audio(random.choice(self.background_files))

            # Add rir
            if self.rir_files is not None and random.random() < self.rir_prob:
                rir = load_audio(random.choice(self.rir_files))

            # Add codec
            if random.random() < self.codecs_prob:
                codec = random.choice(self.codecs)
            
            # Add effect
            if self.effects is not None and random.random() < self.effects_prob:
                effect = _resolve(self.effects)

            # Create effector
            if effect is not None or codec is not None:
                args = {}
                if effect is not None:
                    args['effect'] = effect
                if codec is not None:
                    args.update(codec)
                effector = torchaudio.io.AudioEffector(**args)

            # Do synthesizing
            return synthesize_sample(DATASET_SAMPLE_LENGTH, 
                                           # Effect
                                           effector=effector,
                                           
                                           # Background
                                           background=background, 
                                           background_snr=background_snr,

                                           # Clean voice
                                           clean=clean,
                                           clean_treshold=self.speech_treshold,
                                           clean_smooth=self.speech_smooth,
                                           clean_tempo=speech_tempo,

                                           # RIR
                                           rir=rir)

def synthetic_dataset(count, speech_dir = None, background_dir = None, rir_dir = None):
    return SyntheticDataset(count, speech_dir, background_dir, rir_dir)