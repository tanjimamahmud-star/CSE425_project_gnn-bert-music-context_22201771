import librosa
import numpy as np

def load_and_resample(file_path, target_sr=22050):
    """
    Load audio and resample to target_sr.
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_features(y, sr, feature_type='log_mel', n_mels=128, n_chroma=12):
    """
    Extract normalized features per track.
    feature_type: 'log_mel', 'chroma', or 'mfcc'
    """
    if feature_type == 'log_mel':
        # 128 bins
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
        features = librosa.power_to_db(mel, ref=np.max)
    elif feature_type == 'chroma':
        # 12 bins
        features = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    elif feature_type == 'mfcc':
        features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    else:
        raise ValueError("Unsupported feature type. Use 'log_mel', 'chroma', or 'mfcc'.")

    # Normalize per track (Standardization)
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features

def segment_audio(y, sr, segment_length_sec=5.0):
    """
    Split audio into fixed-length windows.
    Returns a list of audio arrays.
    """
    segment_samples = int(segment_length_sec * sr)
    
    segments = []
    # Loop over the audio and extract fixed windows
    for start in range(0, len(y), segment_samples):
        segment = y[start:start + segment_samples]
        
        # If the last segment is too short, we can pad it or discard it
        # Here we pad with zeros to ensure consistent sizes
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
            
        segments.append(segment)
        
    return segments

def get_segmented_features(file_path, feature_type='log_mel', segment_length_sec=5.0):
    """
    Wrapper function: loads audio, segments it, and extracts features for each segment.
    Returns: numpy array of shape (num_segments, feature_dim, time_frames)
    """
    y, sr = load_and_resample(file_path)
    if y is None:
        return None
        
    segments = segment_audio(y, sr, segment_length_sec)
    
    segment_features = []
    for seg in segments:
        feat = extract_features(seg, sr, feature_type=feature_type)
        segment_features.append(feat)
        
    return np.stack(segment_features) # (num_segments, features, time)
