import random

import soundfile as sf
import numpy as np


class DataLoader:

    @staticmethod
    def load_audio(filename: str, max_frames: int = 300, num_eval=10):

        max_audio = max_frames * 160 + 240
        
        audio, sample_rate = sf.read(filename)

        audiosize = audio.shape[0]

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio = np.pad(audio, (0, shortage), 'wrap')
            audiosize = audio.shape[0]

        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])

        feats = []
        
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

        feat = np.stack(feats, axis=0).astype(np.float64)

        return feat


if __name__ == "__main__":
    f = DataLoader.load_audio("../samples/dr-uthaya-e1.mp3", 160)
    print(f)
