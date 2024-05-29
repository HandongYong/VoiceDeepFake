########################################################
#sampleling rate 44100
# all of voice data length 4s   <need to change>
#All audio data has been standardized to a length of 4 seconds. For data shorter than 4 seconds, repetition was applied, 
# while data longer than 4 seconds was truncated."
########################################################

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Setting input and output directories
input_directory = '  '
output_directory = '  '

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Length of each segment (in seconds)
segment_length = 4.0

# Function to convert audio files to CQT images
def audio_to_cqt_image(audio_path, output_dir, segment_length):
    y, sr = librosa.load(audio_path, sr=44100)  # Load with a sampling rate of 44.1kHz
    end_sample = int(segment_length * sr)  # Number of samples corresponding to 4 seconds
    segment = y[:end_sample]

    # Handling for repetition
    if len(segment) < end_sample:
        repeat_factor = int(np.ceil(end_sample / len(segment)))
        segment = np.tile(segment, repeat_factor)[:end_sample]

    # Calculate CQT spectrum
    cqt = librosa.cqt(segment, sr=sr)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    # Save CQT image
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(cqt_db, sr=sr, x_axis=None, y_axis=None, cmap='viridis')
    plt.axis('off')  # Remove axes
    base_filename = os.path.basename(audio_path).split('.')[0]
    output_filename = os.path.join(output_dir, f'{base_filename}.jpg')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f'Saved CQT image: {output_filename}')

# Process all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.wav'):
        audio_path = os.path.join(input_directory, filename)
        audio_to_cqt_image(audio_path, output_directory, segment_length)

print("Conversion completed.")
