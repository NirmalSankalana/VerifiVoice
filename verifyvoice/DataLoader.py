import random
import webrtcvad
import soundfile as sf
import numpy as np
import librosa
import struct

class DataLoader:

    @staticmethod
    def load_audio(filename: str, max_frames: int = 300, evalmode=True, num_eval=10):

        max_audio = max_frames * 160 + 240
        
        audio, sample_rate = sf.read(filename)


        # Create a VAD instance
        vad = webrtcvad.Vad()
        vad.set_mode(3)  # Set the aggressiveness mode (0-3)

        # Initialize an array to store the speech frames
        speech_frames = []

        # Process the audio in 30ms frames with a 10ms step
        frame_duration = 0.03  # 30ms
        frame_stride = 0.01  # 10ms
        frames = librosa.util.frame(audio, frame_length=int(frame_duration * sample_rate),
                                hop_length=int(frame_stride * sample_rate))
        
        for frame in frames.T:
            # print(f"{frame.shape=}")
            frame_bytes = struct.pack(f"{len(frame)}h", *np.rint(frame * 32767).astype(np.int16))
            if vad.is_speech(frame_bytes, sample_rate):
                speech_frames.append(frame)

        # Concatenate the speech frames into a single array
        speech_data = np.concatenate(speech_frames)
        # speech_data = audio
        audiosize = speech_data.shape[0]

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            speech_data = np.pad(speech_data, (0, shortage), 'wrap')
            audiosize = speech_data.shape[0]

        if evalmode:
            startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
        else:
            startframe =  np.array(
                [np.int64(random.random()*(audiosize-max_audio))])

        feats = []
        if evalmode and max_frames == 0:
            feats.append(speech_data)
        else:
            for asf in startframe:
                feats.append(speech_data[int(asf):int(asf)+max_audio])

        feat = np.stack(feats, axis=0).astype(np.float64)
        print(f"{feat.shape=}")
        return feat


if __name__ == "__main__":
    f = DataLoader.load_audio("../samples/dr-uthaya-e1.mp3", 160)
    print(f)
