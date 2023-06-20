#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Template Author: Jan Schlüter
Onset, Tempo and Beat Detection Author: Christoph Pfleger
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter
import librosa
try:
    import tqdm
except ImportError:
    tqdm = None

import torch
from tempocnn import TempoCNN

def opts_parser():
    usage =\
"""Detects onsets, beats and tempo in WAV files.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('indir',
            type=str,
            help='Directory of WAV files to process.')
    parser.add_argument('outfile',
            type=str,
            help='Output JSON file to write.')
    parser.add_argument('mfs',
            type=int,
            help='SuperFlux max filter size.')
    for i in range(1,5):
        parser.add_argument(f'w{i}',
                type=int,
                help=f'SuperFlux w{i} parameter.')
    parser.add_argument('w5',
            type=float,
            help='SuperFlux w5 parameter.')
    parser.add_argument('delta',
            type=float,
            help='SuperFlux delta parameter.')
    parser.add_argument('pulse_width',
            type=int,
            help='Pulse width for beat detection.')
    parser.add_argument('multiplier',
            type=float,
            help='Correlation multiplier for deciding tempo in beat detection.')
    parser.add_argument('--plot',
            action='store_true',
            help='If given, plot something for every file processed.')
    return parser


def detect_everything(filename, options):#, tempo_model):
    """
    Computes some shared features and calls the onset, tempo and beat detectors.
    """
    # read wave file (this is faster than librosa.load)
    sample_rate, signal = wavfile.read(filename)

    # convert from integer to float
    if signal.dtype.kind == 'i':
        signal = signal / np.iinfo(signal.dtype).max

    # convert from stereo to mono (just in case)
    if signal.ndim == 2:
        signal = signal.mean(axis=-1)

    # compute spectrogram with given number of frames per second
    # hop_length_s = 0.025 # 25ms
    # window_length_s = 0.01 # 10ms
    fps = 70
    hop_length = sample_rate // fps
    spect = librosa.stft(
            signal, n_fft=2048, hop_length=hop_length, window='hann')

    # only keep the magnitude
    magspect = np.abs(spect)

    # TODO: Impplement max filter for SuperFlux
    magspect = maximum_filter(magspect, options.mfs)

    # compute a mel spectrogram
    melspect = librosa.feature.melspectrogram(
            S=magspect, sr=sample_rate, n_mels=80, fmin=27.5, fmax=8000)

    # compress magnitudes logarithmically
    melspect = np.log1p(100 * melspect) 

    # compute onset detection function
    odf, odf_rate = onset_detection_function(
            sample_rate, signal, fps, spect, magspect, melspect, options)

    # detect onsets from the onset detection function
    onsets = detect_onsets(odf_rate, odf, options)

    # detect tempo from everything we have
    tempo = detect_tempo(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, options)#, tempo_model)

    # detect beats from everything we have (including the tempo)
    beats = detect_beats(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, tempo, options)

    # plot some things for easier debugging, if asked for it
    if options.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, sharex=True)
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(filename)
        axes[0].set_title('melspect')
        axes[0].imshow(melspect, origin='lower', aspect='auto',
                       extent=(0, melspect.shape[1] / fps,
                               -0.5, melspect.shape[0] - 0.5))
        axes[1].set_title('onsets')
        axes[1].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in onsets:
            axes[1].axvline(position, color='tab:orange')
        axes[2].set_title('beats (tempo: %r)' % list(np.round(tempo, 2)))
        axes[2].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in beats:
            axes[2].axvline(position, color='tab:red')
        plt.show()

    return {'onsets': list(np.round(onsets, 3)),
            'beats': list(np.round(beats, 3)),
            'tempo': list(np.round(tempo, 2))}


def H(x):
    return np.maximum(x, 0)


def onset_detection_function(sample_rate, signal, fps, spect, magspect,
                             melspect, options):
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """
    # # we only have a dumb dummy implementation here.
    # # it returns every 1000th absolute sample value of the input signal.
    # # this is not a useful solution at all, just a placeholder.
    # values = np.abs(signal[::1000])
    # values_per_second = sample_rate / 1000

    values = np.sum(H(melspect[:,1:] - melspect[:,:-1]), axis=0)
    values -= values.min()
    values /= values.max()
    values_per_second = fps

    return values, values_per_second


def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    # # we only have a dumb dummy implementation here.
    # # it returns the timestamps of the 100 strongest values.
    # # this is not a useful solution at all, just a placeholder.
    # strongest_indices = np.argpartition(odf, 100)[:100]
    # strongest_indices.sort()

    w1 = options.w1
    w2 = options.w2
    w3 = options.w3
    w4 = options.w4
    w5 = options.w5 * odf_rate
    delta = options.delta
    onset_indices = [0]
    for i in range(max(w1, w3), len(odf)-max(w2, w4)):
        local_max = odf[i-w1:i+w2+1].max()
        local_mean = odf[i-w3:i+w4+1].mean()
        if np.isclose(odf[i], local_max) and odf[i] >= local_mean + delta and i - onset_indices[-1] > w5:
            onset_indices.append(i)
    return np.array(onset_indices[1:]) / odf_rate

def autocorr(x, fps=70, min_bpm=60, max_bpm=200):
    result = np.correlate(x, x, mode='same')
    mid = int(np.ceil(len(result)/2))
    min_lag = int(60*fps/max_bpm)
    max_lag = int(60*fps/min_bpm)
    return result[mid+min_lag:mid+max_lag]

def get_peak(x, fps=70, max_bpm=200):
    min_lag = int(60*fps/max_bpm)
    return 60*fps/(np.argmax(x)+min_lag)

def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, options):#, tempo_model):
    """
    Detect tempo using any of the input representations.
    Returns one tempo or two tempo estimations.
    """    
    
    tempo = float(get_peak(autocorr(odf, odf_rate), odf_rate))
    return [tempo/2, tempo]


def make_pulse_train(tempo, phase=0, length=70*3, pulse_width=5, fps=70):
    return (np.arange(-phase,length-phase) % (60*fps/tempo) < pulse_width).astype(float)

def get_phase_and_correlation(odf, tempo, pulse_width=5, fps=70):
    filtered_odf = np.array(odf.copy())
    #filtered_odf[filtered_odf < np.median(filtered_odf)] = 0
    corr = np.correlate(filtered_odf, make_pulse_train(tempo, length=len(odf), pulse_width=pulse_width, fps=fps), 'same')
    mid = int(np.ceil(len(corr)/2))
    phase = corr[mid:].argmax()
    return phase, corr[mid:][phase]

def choose_tempo_and_phase(odf, tempi, pulse_width=5, fps=70, multiplier=2):
    phase1, corr1 = get_phase_and_correlation(odf, tempi[0], pulse_width=pulse_width, fps=fps)
    phase2, corr2 = get_phase_and_correlation(odf, tempi[1], pulse_width=pulse_width, fps=fps)
    if corr2 > multiplier * corr1:
        return tempi[1], phase2
    return tempi[0], phase1

def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempi, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    pulse_width = options.pulse_width
    tempo, phase = choose_tempo_and_phase(odf, tempi, pulse_width, fps, options.multiplier)
    
    threshold = 60/tempo/3
    onsets = np.asarray(onsets)
    beats = []
    for beat in np.arange(phase, len(odf), (60*fps/tempo)):
        beat_middle = (beat+pulse_width/2)/fps
        if len(onsets) != 0:
            nearest_id = np.abs(onsets - beat_middle).argmin()
            nearest = onsets[nearest_id]
        else:
            nearest = -100
        if np.abs(nearest - beat_middle) < threshold:
            beats.append(nearest)
        else:
            beats.append(beat_middle)
    return beats


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    
    print('using options:', str(options))

    #tempo_model = TempoCNN(256//options.resolution)
    #tempo_model.load_state_dict(torch.load('tempo_model_3.pt'))
    #tempo_model.eval()
    #tempo_model = tempo_model.cuda()
    
    # iterate over input directory
    indir = Path(options.indir)
    infiles = list(indir.glob('*.wav'))
    if tqdm is not None:
        infiles = tqdm.tqdm(infiles, desc='File')
    results = {}
    for filename in infiles:
        results[filename.stem] = detect_everything(filename, options)#, tempo_model)

    # write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

