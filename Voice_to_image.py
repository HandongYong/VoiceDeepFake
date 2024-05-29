
#### need to import librosa
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


################################################
# Change image by cqt #
################################################

def audio_to_cqt_image(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=44100)
    cqt = librosa.cqt(y, sr=sr)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q Power Spectrogram')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

audio_to_cqt_image('input_audio.wav', 'cqt_spectrogram.png')


################################################
# power log spectrogram #
################################################

def audio_to_power_log_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=44100)
    D = np.abs(librosa.stft(y))**2
    S = librosa.power_to_db(D, ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power Log Spectrogram')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

audio_to_power_log_spectrogram('input_audio.wav', 'power_log_spectrogram.png')


################################################
# mel spectrogram spectrogram #
################################################

def audio_to_mel_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=44100)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Frequency Spectrogram')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

audio_to_mel_spectrogram('input_audio.wav', 'mel_spectrogram.png')
