"""
Embedding providers for semantic search.

Provides vector embeddings for memory entries to enable
semantic similarity search.
"""

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for text.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate similarity between two embedding vectors.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        pass


class SimpleEmbedding(EmbeddingProvider):
    """
    Simple TF-IDF-like embedding provider.
    
    This is a lightweight alternative that doesn't require
    external libraries. For better semantic search, use
    SentenceTransformerEmbedding when available.
    
    Features:
    - Word frequency based embeddings
    - Cosine similarity for matching
    - Fast and lightweight (< 100ms for retrieval)
    - No external dependencies
    """
    
    # Embedding dimension (hash-based)
    DIMENSION = 128
    
    def __init__(self, dimension: int = 128):
        """
        Initialize the embedding provider.
        
        Args:
            dimension: Embedding vector dimension
        """
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension
    
    def embed(self, text: str) -> List[float]:
        """
        Generate a simple embedding for text.
        
        Uses a hash-based approach to create a fixed-dimension
        vector that captures word presence information.
        """
        if not text:
            return [0.0] * self._dimension
        
        # Tokenize and normalize
        words = self._tokenize(text)
        
        # Create embedding vector
        vector = [0.0] * self._dimension
        
        for word in words:
            # Hash word to get index
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            index = word_hash % self._dimension
            
            # Use a second hash for the sign
            sign_hash = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            sign = 1 if sign_hash % 2 == 0 else -1
            
            # Add contribution (TF-like weighting)
            vector[index] += sign * (1.0 / len(words))
        
        # L2 normalize
        return self._normalize(vector)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Returns a value between 0 and 1, where 1 is most similar.
        """
        if not vec1 or not vec2:
            return 0.0
        
        if len(vec1) != len(vec2):
            return 0.0
        
        # Dot product
        dot = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        # Cosine similarity, normalized to [0, 1]
        cosine = dot / (mag1 * mag2)
        return (cosine + 1) / 2
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split
        words = []
        current_word = []
        
        for char in text:
            if char.isalnum():
                current_word.append(char)
            elif current_word:
                words.append(''.join(current_word))
                current_word = []
        
        if current_word:
            words.append(''.join(current_word))
        
        # Filter short words
        words = [w for w in words if len(w) > 2]
        
        return words
    
    def _normalize(self, vector: List[float]) -> List[float]:
        """L2 normalize a vector."""
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return vector
        return [x / magnitude for x in vector]


class SentenceTransformerEmbedding(EmbeddingProvider):
    """
    Sentence-transformers based embedding provider.
    
    Provides high-quality semantic embeddings using pre-trained
    transformer models. Requires sentence-transformers library.
    """
    
    # Default model for efficiency
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize with a sentence-transformers model.
        
        Args:
            model_name: Name of the model to use. Defaults to MiniLM.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._dimension = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using simple embeddings instead."
                )
                raise
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            _ = self.model  # Force load
        return self._dimension or 384  # Default for MiniLM
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding using sentence-transformers."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        try:
            from sentence_transformers import util
            import numpy as np
            
            score = util.cos_sim(np.array(vec1), np.array(vec2))
            # Normalize to [0, 1]
            return (float(score) + 1) / 2
        except ImportError:
            # Fall back to manual calculation
            return SimpleEmbedding().similarity(vec1, vec2)


def get_embedding_provider(
    use_transformers: bool = False,
    model_name: Optional[str] = None,
) -> EmbeddingProvider:
    """
    Get an embedding provider.
    
    Args:
        use_transformers: Whether to use sentence-transformers
        model_name: Model name for sentence-transformers
        
    Returns:
        An embedding provider instance
    """
    if use_transformers:
        try:
            return SentenceTransformerEmbedding(model_name)
        except ImportError:
            logger.warning("Falling back to simple embeddings")
    
    return SimpleEmbedding()
