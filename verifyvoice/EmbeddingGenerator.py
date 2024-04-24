class EmbeddingGenerator():
    def __init__(self, model_name:str):
        self.model_name = model_name

    def get_embedding(self, audio):
        return self.model_name(audio)