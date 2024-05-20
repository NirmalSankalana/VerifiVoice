from sklearn.metrics.pairwise import cosine_similarity

class Similarity:
    """
    Similarity class for calculating similarity metrics.

    cosine_similarity:
    Calculate the cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Mean of the cosine similarity between the two vectors.
    """

    
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
    
        return cosine_similarity(a, b).mean()
