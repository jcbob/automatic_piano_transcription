import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e10
from data_preprocessing import get_train_data
from data_preprocessing import group_cqt_frames

def preprocess_and_pickle_data(root_folder, output_pickle_file, sampling_rate, hop_length, n_bins, show_cqt_pr, pr_in_frames, cqt_in_frames, num_frames_before, num_frames_after):
    count = 0
    cqt_array = np.empty((1, 88, 7))
    label_array = np.empty((1, 88))

    cqt_list = []
    label_list = []

    # Iterate over all subdirectories in the root folder
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.wav'):
                # print(f"processing audio and midi files - {file}")
                audio_path = os.path.join(subdir, file)
                midi_path = os.path.join(subdir, file[:-4] + '.mid')  # Assuming MIDI files have the same name as WAV files

                # Preprocess the data for the current pair of audio and MIDI files
                cqt, labels = get_train_data(audio_path, sampling_rate, hop_length, n_bins, midi_path, show_cqt_pr, pr_in_frames, cqt_in_frames)
                labels = np.transpose(labels)
                cqt = group_cqt_frames(cqt, num_frames_before, num_frames_after)
                cqt = np.abs(cqt)
                # print("got labels and cqt")
                # print(f"frames - {len(labels)}")

                # Append the preprocessed data to the list
                cqt_list.extend(cqt)
                label_list.extend(labels)
                
                # print("appended cqt and labels")
                count += 1
                print(f"files processed - {count}")
        # return cqt
                
    cqt_array = np.array(cqt_list)
    label_array = np.array(label_list)
    # return cqt_array, label_array
    # Save the list of preprocessed data to a single pickle file
    with open(output_pickle_file, 'wb') as f:
        pickle.dump((cqt_array, label_array), f)