import random

import soundfile as sf
import numpy as np

class DataLoader:

    @staticmethod
    def load_audio(filename: str, max_frames: int, num_eval=10):
        max_audio = max_frames * 160 + 240

        # Read wav file and convert to torch tensor
        audio, sample_rate = sf.read(filename)

        audiosize = audio.shape[0]

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio = np.pad(audio, (0, shortage), 'wrap')
            audiosize = audio.shape[0]

        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)

        feats = []
        if max_frames == 0:
            feats.append(audio)
        else:
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])

        feat = np.stack(feats, axis=0).astype(np.float64)

        return feat


if __name__ == "__main__":
    f = DataLoader.load_audio("../samples/dr-uthaya-e1.mp3", 160)
    print(f)
