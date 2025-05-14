import os
import mne

directory = os.getcwd()
prject_name = os.path.basename(directory)

def find_eeg_files(directory):
    eeg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.EEG'):
                eeg_files.append(os.path.join(root, file))
    return eeg_files

def convert_and_remove_eeg(eeg_files):
    for eeg_file_path in eeg_files:
        base, _ = os.path.splitext(eeg_file_path)
        edf_file_path = base + '.edf'
        try:
            raw = mne.io.read_raw_nihon(eeg_file_path, preload=True)
            raw.export(edf_file_path, fmt='edf', overwrite=True)
            print(f"Converted {eeg_file_path} to {edf_file_path}")
            os.remove(eeg_file_path)
            print(f"Deleted original EEG file: {eeg_file_path}")
        except Exception as e:
            print(f"Failed to convert {eeg_file_path}: {e}")

if __name__ == "__main__":
    eeg_files = find_eeg_files(directory)
    convert_and_remove_eeg(eeg_files)