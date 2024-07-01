import numpy as np
import os
import librosa

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
                
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths

def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

# set the audio into the format that can be fed into the model
# resample -> convert to mono -> output the audio  
# track [n_sample, n_channel]
def prepprocess_audio(track, ofs, rfs, mono_type = "mix"):
    if track.shape[-1] > 1:
        # stereo
        if mono_type == "mix":
            track = np.transpose(track, (1,0))
            track = librosa.to_mono(track)
        elif mono_type == "left":
            track = track[:, 0]
        elif mono_type == "right":
            track = track[:, 1]
    else:
        track = track[:, 0]
    # track [n_sample]
    if ofs != rfs:
        track = librosa.resample(track, orig_sr=ofs, target_sr=rfs)
    return track
