#!/usr/bin/env python3
"""
Script to display details about an EDF file
"""
import sys
import os
import numpy as np

def format_duration(seconds):
    """Format duration in a human-readable way"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def read_edf_with_mne(file_path):
    """Read EDF file using MNE with error handling"""
    try:
        import mne
    except ImportError:
        print("Error: mne is not installed. Please install it with: pip install mne")
        sys.exit(1)
    
    # Try different approaches to read the EDF file
    read_attempts = [
        {'preload': True, 'encoding': 'latin1', 'verbose': False},
        {'preload': False, 'encoding': 'latin1', 'verbose': False},
        {'preload': True, 'verbose': False},
        {'preload': False, 'verbose': False},
        {'preload': True, 'encoding': 'latin1', 'exclude': [], 'verbose': False},
        {'preload': True, 'encoding': 'latin1', 'stim_channel': None, 'verbose': False},
        {'preload': True, 'encoding': 'latin1', 'exclude': [], 'stim_channel': None, 'verbose': False},
    ]
    
    raw = None
    last_error = None
    
    for i, params in enumerate(read_attempts):
        try:
            raw = mne.io.read_raw_edf(file_path, **params)
            print('------------------------------------------------------------------------------------------')
            print(f"Successfully read EDF file with parameters: {params} i = {i+1}/{len(read_attempts)} i: {i}")
            if not params.get('preload', True):
                raw.load_data()
            break
        except Exception as e:
            last_error = e
            if i < len(read_attempts) - 1:
                continue
            else:
                # Try pyedflib as fallback
                try:
                    import pyedflib
                    edf = pyedflib.EdfReader(file_path)
                    n_channels = edf.signals_in_file
                    ch_names = [edf.getSignalLabel(i) for i in range(n_channels)]
                    sfreqs = [edf.getSampleFrequency(i) for i in range(n_channels)]
                    sfreq = sfreqs[0] if sfreqs else 256
                    if len(set(sfreqs)) > 1:
                        from collections import Counter
                        sfreq = Counter(sfreqs).most_common(1)[0][0]
                    
                    data = np.array([edf.readSignal(i) for i in range(n_channels)])
                    ch_types = ['misc'] * n_channels
                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                    raw = mne.io.RawArray(data, info)
                    edf.close()
                    break
                except Exception:
                    raise last_error
    
    if raw is None:
        raise ValueError(f"Failed to read EDF file: {last_error}")
    
    metadata = {
        'file_name': os.path.basename(file_path),
        'start_date': raw.info['meas_date'],
        'duration_sec': raw.times[-1],
        'duration_min': raw.times[-1] / 60,
        'number_of_signals': len(raw.ch_names),
        'signal_labels': raw.ch_names,
        'sampling_frequency': raw.info['sfreq'],
        'highpass': raw.info['highpass'],
        'lowpass': raw.info['lowpass'],
        'annotations': raw.annotations if raw.annotations else None,
        'n_samples': raw.n_times,
        'bad_channels': raw.info['bads']
    }
    return metadata, raw

def main():
    edf_path = "EDF_Format/The_Human_Sleep_Project/sub-I0006179000263/ses-1/eeg/SUB-I0006179000263_SES-1_TASK-PSG_EEG.EDF"
    
    print(f"Reading EDF file: {edf_path}")
    print("=" * 80)
    
    if not os.path.exists(edf_path):
        print(f"❌ Error: File not found: {edf_path}")
        sys.exit(1)
    
    try:
        metadata, raw = read_edf_with_mne(edf_path)
        
        print("\n📁 FILE INFORMATION")
        print("-" * 80)
        print(f"File name: {metadata['file_name']}")
        print(f"File path: {edf_path}")
        file_size = os.path.getsize(edf_path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        
        print("\n⏱️  DURATION")
        print("-" * 80)
        print(f"Duration: {format_duration(metadata['duration_sec'])}")
        print(f"Duration (seconds): {metadata['duration_sec']:.2f} s")
        print(f"Duration (minutes): {metadata['duration_min']:.2f} min")
        print(f"Duration (hours): {metadata['duration_min'] / 60:.2f} hours")
        
        print("\n📊 RECORDING INFORMATION")
        print("-" * 80)
        print(f"Start date/time: {metadata.get('start_date', 'N/A')}")
        print(f"Number of channels/electrodes: {metadata['number_of_signals']}")
        print(f"Sampling frequency: {metadata['sampling_frequency']} Hz")
        print(f"Total samples: {metadata['n_samples']:,}")
        print(f"Highpass filter: {metadata.get('highpass', 'N/A')} Hz")
        print(f"Lowpass filter: {metadata.get('lowpass', 'N/A')} Hz")
        
        print("\n🔌 ELECTRODES/CHANNELS")
        print("-" * 80)
        electrodes = metadata['signal_labels']
        print(f"Total electrodes: {len(electrodes)}")
        print("\nElectrode list:")
        for i, electrode in enumerate(electrodes, 1):
            print(f"  {i:3d}. {electrode}")
        
        if metadata.get('bad_channels'):
            print(f"\n⚠️  Bad channels: {metadata['bad_channels']}")
        else:
            print("\n✓ No bad channels marked")
        
        if metadata.get('annotations'):
            print(f"\n📝 Annotations: {len(metadata['annotations'])} annotation(s) found")
        else:
            print("\n📝 Annotations: None")
        
        print("\n" + "=" * 80)
        print("Summary complete!")
        
    except Exception as e:
        print(f"\n❌ Error reading EDF file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
