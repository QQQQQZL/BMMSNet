import glob

import numpy as np


# concat the mel-spectogram and speech envelope
# mel (T, 10) + envelope (T, 1) repeat 5 times -> MelEnv (T, 5)

# change the file path according to your experiment
mel_files = glob.glob('/data/qiuzelin/Data/EEG/SPARRKULEE/derivatives/split_data/*mel.*')
print(len(mel_files))

for idx, info in enumerate(mel_files):
    if idx % 100 == 0:
        print(idx)
    mel = np.load(info)
    env= np.load(info.replace('mel','envelope'))
    env = np.tile(env, (1, 5))
    
    melenv = np.concatenate((mel, env), 1).astype(np.float32)
    
    np.save(info.replace('mel','MelEnv'), melenv)
    


