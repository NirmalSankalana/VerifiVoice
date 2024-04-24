import random

import soundfile
import numpy as np


class DataLoader:
    @staticmethod
    def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

        # Maximum audio length
        max_audio = max_frames * 160 + 240

        # Read wav file and convert to torch tensor
        audio, sample_rate = soundfile.read(filename)

        audiosize = audio.shape[0]

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio = np.pad(audio, (0, shortage), 'wrap')
            audiosize = audio.shape[0]

        if evalmode:
            startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
        else:
            startframe = np.array(
                [np.int64(random.random()*(audiosize-max_audio))])

        feats = []
        if evalmode and max_frames == 0:
            feats.append(audio)
        else:
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])

        feat = np.stack(feats, axis=0).astype(np.float64)

        return feat