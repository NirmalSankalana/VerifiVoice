from sklearn.metrics.pairwise import cosine_similarity

class Similarity:
    
        
    @staticmethod
    def cosine_similarity(a, b):
        """
        Calculate the cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Mean of the cosine similarity between the two vectors.
        """
        similarity = cosine_similarity(a, b)
        # print(f"{similarity=} {similarity.shape=}")
        return similarity.mean()
