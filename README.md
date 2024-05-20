
# VerifyVoice: Python Package

The VerifyVoice library is designed for text-independent speaker verification using voice embeddings. It provides an easy-to-use interface for speaker verification by loading a pre-trained model and extracting embeddings from audio files. You can compare audio files to determine if they belong to the same speaker by computing similarity scores between these embeddings. This documentation provides a basic overview and example usage to help you get started with the library.

## Installation
To install the VerifyVoice package, use the following command:

```bash
pip install verifyvoice
```

## Loading the Models
To use the VerifyVoice library, you need to load a pre-trained model using the `ModelLoader` class. Here is an example of how to initialize the model loader with a specific model:

```python
from verifyvoice import ModelLoader 
model_loader = ModelLoader(model_name="WavLM", attention_heads=8)
```

You can select different pre-trained models trained with different attention heads (4, 8, 16). If your local machine does not have the models, they will automatically download from Hugging Face and save to the cache folder.

## Extracting Embeddings
Once the model is loaded, you can extract embeddings from audio files. These embeddings represent the audio features used for comparison:

```python
audio_path = '/path/to/audio.wav'
embedding = model_loader.get_embedding(audio_path)
```

## Preprocessing Audio Data
The `DataLoader` class is responsible for loading and preprocessing audio data, including noise reduction and voice activity detection (VAD).

```python
from verifyvoice import DataLoader

processed_audio = DataLoader.load_audio(audio_path, max_frames=160, evalmode=True, num_eval=10, vad_mode=True)
```

## Threshold
The optimal threshold value for the currently loaded model is based on the number of attention heads used in the model configuration. Threshold values are critical in speaker verification tasks as they determine the decision boundary for classifying whether two audio samples belong to the same speaker or not.

```python
threshold = model_loader.get_threshold()
```

## Comparing Embeddings
To verify if two audio files belong to the same speaker, you can compute the cosine similarity between their embeddings. A higher cosine similarity score indicates higher similarity between the embeddings:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity between embeddings
similarity_1_4 = cosine_similarity(embedding1, embedding4).mean()
similarity_2_4 = cosine_similarity(embedding2, embedding4).mean()
similarity_4_1 = cosine_similarity(embedding4, embedding1).mean()

if similarity_1_4 >= threshold:
    print("same speaker")
else: 
    print("different speaker")
```

By following these steps, you can effectively use the VerifyVoice library for speaker verification tasks.
