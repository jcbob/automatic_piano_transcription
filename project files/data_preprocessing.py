import librosa
import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e10
import numpy as np
import matplotlib.pyplot as plt

def get_train_data(audio, sampling_rate, hop_length, n_bins, midi, show_true, pr_in_frames=False, cqt_in_frames=False):
    # print("getting data...")

    # get labels
    pm = pretty_midi.PrettyMIDI(midi)
    frame_rate = sampling_rate/hop_length
    labels = create_labels(pm, frame_rate)
    
    # get CQT
    y, sampling_rate = librosa.load(audio, sr=sampling_rate)
    cqt = librosa.cqt(y, sr=sampling_rate, hop_length=hop_length, n_bins=n_bins)
    
    # trim CQT to midi file
    midi_duration = pm.get_end_time()
    cqt = trim_cqt_to_midi_length(cqt, midi_duration, frame_rate)

    # cqt = np.abs(cqt)

    # print(f"cqt time - {(cqt.shape[1]*512)/sampling_rate}")
    # print(f"midi time - {pm.get_end_time()}")
    # print(f"cqt frames - {cqt.shape[1]}")
    # print(f"midi frames - {len(labels[0])}")
    if show_true:
        plot_piano_roll_and_cqt(cqt, sampling_rate, hop_length, labels, pm, pr_in_frames, cqt_in_frames)

    # print("done")
    return cqt, labels


def create_labels(pm, frame_rate):
    num_frames = int(pm.get_end_time() * frame_rate)
    # print(num_frames)
    labels = np.zeros((88, num_frames), dtype=int)

    for frame_idx in range(num_frames):
        frame_time_start = frame_idx/frame_rate
        for note in pm.instruments[0].notes:
            if note.start <= frame_time_start < note.end:
                note_index = note.pitch - 21
                labels[note_index, frame_idx] = 1
    
    return labels


def trim_cqt_to_midi_length(cqt, midi_duration, frame_rate):
    num_frames_midi = int(midi_duration * frame_rate)
    trimmed_cqt = cqt[:, :num_frames_midi]
    return trimmed_cqt


def plot_piano_roll_and_cqt(cqt, sampling_rate, hop_length, labels, pm, pr_in_frames=False, cqt_in_frames=False):
    num_frames = labels.shape[1]
    frame_rate = sampling_rate / hop_length
    duration_in_seconds = num_frames / frame_rate

    mag_CQT = np.abs(cqt)
    log_mag_CQT = librosa.amplitude_to_db(mag_CQT)

    

    # Subplot for CQT Spectrogram
    plt.figure(figsize=(20, 5))
    if cqt_in_frames:
        librosa.display.specshow(log_mag_CQT, sr=sampling_rate, hop_length=hop_length, x_axis='frames', y_axis='cqt_note', cmap='magma')
        plt.xlabel('Frames')
    else:
        librosa.display.specshow(log_mag_CQT, sr=sampling_rate, hop_length=hop_length, x_axis='time', y_axis='cqt_note', cmap='magma')
        plt.xlabel('Time [s]')
    plt.colorbar(format='%+2.0f dB')
    plt.title('CQT Spectrogram')
    plt.ylabel('Note')
    plt.tight_layout()
    # plt.savefig('cqt.png', dpi=170)
    plt.show()

    # Subplot for Piano Roll
    plt.figure(figsize=(20, 5))
    if not pr_in_frames:
        plt.imshow(labels, cmap='binary', aspect='auto', origin='lower', extent=[0, pm.get_end_time(), 0, 88])
        plt.xlabel('Time [s]')
    else:
        plt.imshow(labels, cmap='binary', aspect='auto', origin='lower')
        plt.xlabel('Frames')
    plt.ylabel('MIDI Note Index (A0 - C8)')
    plt.title('Piano Roll Representation')
    plt.colorbar(label='Note Presence (1=On, 0=Off)')
    plt.tight_layout()
    # plt.savefig('true_piano_roll.png', dpi=170)
    plt.show()


def group_cqt_frames(cqt, num_frames_before=3, num_frames_after=3):
    num_keys, num_frames = cqt.shape
    grouped_cqt_frames = []
    desired_len = num_frames_before + num_frames_after + 1

    for i in range(num_frames):
        start_frame = max(0, i - num_frames_before)
        end_frame = min(num_frames, i + num_frames_after + 1)
        grouped_frame = cqt[:, start_frame:end_frame]

        if end_frame - start_frame <= desired_len:
            frames_to_add_before = max(0, num_frames_before - (i - start_frame))
            frames_to_add_after = max(0, num_frames_after - (end_frame - i - 1))

            if frames_to_add_before > 0:
                zeros_before = np.zeros((num_keys, frames_to_add_before))
                grouped_frame = np.concatenate((zeros_before, grouped_frame), axis=1)
            if frames_to_add_after > 0:
                zeros_after = np.zeros((num_keys, frames_to_add_after))
                grouped_frame = np.concatenate((grouped_frame, zeros_after), axis=1)
        
        grouped_cqt_frames.append(grouped_frame)

    return np.array(grouped_cqt_frames)

