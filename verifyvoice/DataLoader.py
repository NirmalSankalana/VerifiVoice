import random
import webrtcvad
import soundfile as sf
import numpy as np
import librosa
import struct


class DataLoader:
    """
    Load audio data from a file and process it for further analysis.

    Args:
        filename: Path to the audio file.
        max_frames: Maximum number of frames to consider (default is 300).
        evalmode: Flag indicating evaluation mode (default is True).
        num_eval: Number of evaluations to perform (default is 10).
        vad_mode: Flag indicating whether to use VAD mode (default is True).

    Returns:
        Processed audio data ready for analysis.
    """

    @staticmethod
    def load_audio(filename: str, max_frames: int = 300, evalmode=True, num_eval=10, vad_mode=True):
        """
        Load audio data from a file and process it for further analysis.

        Args:
            filename: Path to the audio file.
            max_frames: Maximum number of frames to consider (default is 300).
            evalmode: Flag indicating evaluation mode (default is True).
            num_eval: Number of evaluations to perform (default is 10).
            vad_mode: Flag indicating whether to use VAD mode (default is True).

        Returns:
            Processed audio data ready for analysis.
        """
        max_audio = max_frames * 160 + 240
        
        audio, sample_rate = sf.read(filename)
        # audio = nr.reduce_noise(y=audio, sr=sample_rate)

        if vad_mode:
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
                frame_bytes = struct.pack(f"{len(frame)}h", *np.rint(frame * 32767).astype(np.int16))
                if vad.is_speech(frame_bytes, sample_rate):
                    speech_frames.append(frame)

            # Concatenate the speech frames into a single array
            if speech_frames:
                speech_data = np.concatenate(speech_frames)
            else:
                speech_data = np.array([])
        # else:
        speech_data = audio

        # Handle empty speech data
        if speech_data.size == 0:
            return speech_data

            # if evalmode:
            #     return np.zeros((num_eval, max_audio), dtype=np.float64)
            # else:
            #     return np.zeros((num_eval, max_audio), dtype=np.float64)

        audiosize = speech_data.shape[0]

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            speech_data = np.pad(speech_data, (0, shortage), 'wrap')
            audiosize = speech_data.shape[0]

        if evalmode:
            startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
        else:
            startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])

        feats = []
        if evalmode and max_frames == 0:
            feats.append(speech_data)
        else:
            for asf in startframe:
                feats.append(speech_data[int(asf):int(asf) + max_audio])

        feat = np.stack(feats, axis=0).astype(np.float64)
        return feat

if __name__ == "__main__":
    f = DataLoader.load_audio("../samples/dr-uthaya-e1.mp3", 160)
    print(f)
