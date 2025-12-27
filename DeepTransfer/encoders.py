import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union
from tqdm import tqdm
import asyncio
import threading

class DualEncoder:
    """
    Handles both Content (SPECTER2) and Style (Style-Embedding) encoding.
    Thread-safe for concurrent async usage.
    """

    def __init__(self, content_model_name: str, style_model_name: str):
        print(f"Loading Content Encoder: {content_model_name}")
        self.content_tokenizer = AutoTokenizer.from_pretrained(content_model_name)
        self.content_model = AutoModel.from_pretrained(content_model_name)
        self.content_model.eval()

        # Thread lock for concurrent encoding (PyTorch models are not thread-safe)
        self._lock = threading.Lock()
        
        print(f"Loading Style Encoder: {style_model_name}")
        # Note: AnnaWegmann/Style-Embedding is a SentenceTransformer model usually, 
        # but can be loaded as HuggingFace model. 
        # For simplicity in this project structure, we assume standard HF interface 
        # or SentenceTransformer interface. Let's use standard HF for consistency if possible,
        # but Style-Embedding might be better with SentenceTransformer library.
        # To avoid extra dependencies, let's try HF first. If it's a roberta base, it works.
        try:
            self.style_tokenizer = AutoTokenizer.from_pretrained(style_model_name)
            self.style_model = AutoModel.from_pretrained(style_model_name)
            self.style_model.eval()
            self.use_hf_style = True
        except Exception as e:
            print(f"Standard HF load failed for style model, trying SentenceTransformer: {e}")
            from sentence_transformers import SentenceTransformer
            self.style_model = SentenceTransformer(style_model_name)
            self.use_hf_style = False

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_content(self, texts: Union[str, List[str]], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generates content embeddings using SPECTER2.
        SPECTER2 uses [CLS] token representation.
        Thread-safe for concurrent usage.

        Args:
            texts: Input text(s) to encode
            batch_size: Batch size for encoding (default: 32)
            show_progress: Whether to show progress bar (default: True)

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Acquire lock for thread-safe encoding
        with self._lock:
            # For small batches, encode directly without progress bar
            if len(texts) <= batch_size:
                inputs = self.content_tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

                with torch.no_grad():
                    outputs = self.content_model(**inputs)

                # SPECTER2 uses the last hidden state of the first token (CLS)
                embeddings = outputs.last_hidden_state[:, 0, :]
                return embeddings.cpu().numpy()

            # For large batches, process in chunks with progress bar
            all_embeddings = []

            iterator = range(0, len(texts), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Encoding (Content)", total=(len(texts) + batch_size - 1) // batch_size)

            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                inputs = self.content_tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

                with torch.no_grad():
                    outputs = self.content_model(**inputs)

                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu().numpy())

            return np.vstack(all_embeddings)

    def encode_style(self, texts: Union[str, List[str]], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generates style embeddings.
        Thread-safe for concurrent usage.

        Args:
            texts: Input text(s) to encode
            batch_size: Batch size for encoding (default: 32)
            show_progress: Whether to show progress bar (default: True)

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Acquire lock for thread-safe encoding
        with self._lock:
            if not self.use_hf_style:
                # SentenceTransformer path
                # SentenceTransformer already has progress bar support
                return self.style_model.encode(texts, show_progress_bar=show_progress, batch_size=batch_size)

            # HF Path (assuming RoBERTa-like)
            # For small batches, encode directly without progress bar
            if len(texts) <= batch_size:
                inputs = self.style_tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                with torch.no_grad():
                    outputs = self.style_model(**inputs)

                # Usually Mean Pooling is better for sentence representation if not CLS trained
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                return embeddings.cpu().numpy()

            # For large batches, process in chunks with progress bar
            all_embeddings = []

            iterator = range(0, len(texts), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Encoding (Style)", total=(len(texts) + batch_size - 1) // batch_size)

            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                inputs = self.style_tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

                with torch.no_grad():
                    outputs = self.style_model(**inputs)

                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                all_embeddings.append(embeddings.cpu().numpy())

            return np.vstack(all_embeddings)
