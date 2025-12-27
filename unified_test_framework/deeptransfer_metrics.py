"""
DeepTransfer-Specific Metrics
Implements dual-encoder based metrics for evaluating content-style disentanglement
"""
import numpy as np
import torch
from typing import List, Dict, Union
from tqdm import tqdm


class DualEncoderMetrics:
    """Metrics based on SPECTER2 (content) and Style-Embedding (style) encoders"""

    def __init__(
        self,
        content_model_name: str = "allenai/specter2_base",
        style_model_name: str = "AnnaWegmann/Style-Embedding",
        use_gpu: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize dual encoder metrics.

        Args:
            content_model_name: HuggingFace model for content encoding
            style_model_name: HuggingFace model for style encoding
            use_gpu: Whether to use GPU
            batch_size: Batch size for encoding
        """
        self.batch_size = batch_size
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        print(f"Initializing Dual Encoder Metrics on device: {self.device}")

        # Load content encoder (SPECTER2)
        print(f"Loading content encoder: {content_model_name}")
        from transformers import AutoTokenizer, AutoModel
        self.content_tokenizer = AutoTokenizer.from_pretrained(content_model_name)
        self.content_model = AutoModel.from_pretrained(content_model_name).to(self.device)
        self.content_model.eval()

        # Load style encoder (Style-Embedding)
        print(f"Loading style encoder: {style_model_name}")
        try:
            self.style_tokenizer = AutoTokenizer.from_pretrained(style_model_name)
            self.style_model = AutoModel.from_pretrained(style_model_name).to(self.device)
            self.style_model.eval()
            self.use_hf_style = True
        except Exception as e:
            print(f"Loading as SentenceTransformer: {e}")
            from sentence_transformers import SentenceTransformer
            self.style_model = SentenceTransformer(style_model_name)
            if use_gpu and torch.cuda.is_available():
                self.style_model = self.style_model.to(self.device)
            self.use_hf_style = False

        print("Dual encoders loaded successfully")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode_content(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using content encoder (SPECTER2).

        Args:
            texts: List of text strings

        Returns:
            Numpy array of content embeddings [N, D]
        """
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding content"):
                batch_texts = texts[i:i + self.batch_size]

                inputs = self.content_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)

                outputs = self.content_model(**inputs)

                # SPECTER2 uses [CLS] token
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_style(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using style encoder.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of style embeddings [N, D]
        """
        if not self.use_hf_style:
            # SentenceTransformer path
            print("Encoding style with SentenceTransformer...")
            return self.style_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )

        # HuggingFace path
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding style"):
                batch_texts = texts[i:i + self.batch_size]

                inputs = self.style_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)

                outputs = self.style_model(**inputs)

                # Mean pooling for style embeddings
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of vectors.

        Args:
            vec1: [N, D] array
            vec2: [N, D] array

        Returns:
            [N,] array of similarities
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-9)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-9)

        # Compute cosine similarity
        similarity = np.sum(vec1_norm * vec2_norm, axis=1)

        return similarity

    def calculate_content_preservation(
        self,
        source_texts: List[str],
        transferred_texts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate content preservation using content encoder.
        Measures how well semantic content is preserved after style transfer.

        Args:
            source_texts: Original de-styled texts
            transferred_texts: Style-transferred texts

        Returns:
            Dictionary with content preservation metrics
        """
        print("\n[DeepTransfer] Calculating content preservation...")

        # Encode both
        source_embs = self.encode_content(source_texts)
        transferred_embs = self.encode_content(transferred_texts)

        # Compute similarities
        similarities = self.cosine_similarity(source_embs, transferred_embs)

        return {
            'dt_content_preservation_mean': round(float(np.mean(similarities)), 4),
            'dt_content_preservation_std': round(float(np.std(similarities)), 4),
            'dt_content_preservation_min': round(float(np.min(similarities)), 4),
            'dt_content_preservation_max': round(float(np.max(similarities)), 4)
        }

    def calculate_style_similarity(
        self,
        reference_texts: List[str],
        transferred_texts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate style similarity using style encoder.
        Measures how well the transferred text matches target style.

        Args:
            reference_texts: Target style reference texts
            transferred_texts: Style-transferred texts

        Returns:
            Dictionary with style similarity metrics
        """
        print("\n[DeepTransfer] Calculating style similarity...")

        # Encode both
        reference_embs = self.encode_style(reference_texts)
        transferred_embs = self.encode_style(transferred_texts)

        # Compute similarities
        similarities = self.cosine_similarity(reference_embs, transferred_embs)

        return {
            'dt_style_similarity_mean': round(float(np.mean(similarities)), 4),
            'dt_style_similarity_std': round(float(np.std(similarities)), 4),
            'dt_style_similarity_min': round(float(np.min(similarities)), 4),
            'dt_style_similarity_max': round(float(np.max(similarities)), 4)
        }

    def calculate_content_style_disentanglement(
        self,
        source_texts: List[str],
        reference_texts: List[str],
        transferred_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate content-style disentanglement quality.

        High quality transfer should:
        - High content similarity: source <-> transferred (in content space)
        - High style similarity: reference <-> transferred (in style space)
        - Moderate content difference: reference <-> transferred (in content space)

        Args:
            source_texts: De-styled source texts
            reference_texts: Target style references
            transferred_texts: Transferred texts

        Returns:
            Dictionary with disentanglement metrics
        """
        print("\n[DeepTransfer] Calculating content-style disentanglement...")

        # Encode content space
        print("Encoding in content space...")
        source_content_embs = self.encode_content(source_texts)
        reference_content_embs = self.encode_content(reference_texts)
        transferred_content_embs = self.encode_content(transferred_texts)

        # Encode style space
        print("Encoding in style space...")
        reference_style_embs = self.encode_style(reference_texts)
        transferred_style_embs = self.encode_style(transferred_texts)

        # Content preservation (should be HIGH)
        content_preservation = self.cosine_similarity(
            source_content_embs, transferred_content_embs
        )

        # Style transfer (should be HIGH)
        style_transfer = self.cosine_similarity(
            reference_style_embs, transferred_style_embs
        )

        # Content differentiation (should be MODERATE, not too high)
        # This shows that transferred text is not just copying reference content
        content_diff = self.cosine_similarity(
            reference_content_embs, transferred_content_embs
        )

        # Disentanglement score: balance between preserving content and changing style
        # Good disentanglement = high content preservation + high style transfer + reasonable content diff
        disentanglement_score = (content_preservation + style_transfer) / 2

        return {
            'dt_disentangle_content_preservation_mean': round(float(np.mean(content_preservation)), 4),
            'dt_disentangle_style_transfer_mean': round(float(np.mean(style_transfer)), 4),
            'dt_disentangle_content_diff_mean': round(float(np.mean(content_diff)), 4),
            'dt_disentanglement_score': round(float(np.mean(disentanglement_score)), 4),
            'dt_disentanglement_score_std': round(float(np.std(disentanglement_score)), 4)
        }

    def calculate_dual_space_alignment(
        self,
        source_texts: List[str],
        reference_texts: List[str],
        transferred_texts: List[str],
        content_weight: float = 0.5,
        style_weight: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate dual-space alignment score.
        Weighted combination of content preservation and style similarity.

        Args:
            source_texts: De-styled source texts
            reference_texts: Target style references
            transferred_texts: Transferred texts
            content_weight: Weight for content preservation
            style_weight: Weight for style similarity

        Returns:
            Dictionary with alignment scores
        """
        print("\n[DeepTransfer] Calculating dual-space alignment...")

        # Get content preservation
        source_content_embs = self.encode_content(source_texts)
        transferred_content_embs = self.encode_content(transferred_texts)
        content_sim = self.cosine_similarity(source_content_embs, transferred_content_embs)

        # Get style similarity
        reference_style_embs = self.encode_style(reference_texts)
        transferred_style_embs = self.encode_style(transferred_texts)
        style_sim = self.cosine_similarity(reference_style_embs, transferred_style_embs)

        # Weighted alignment score
        alignment_score = content_weight * content_sim + style_weight * style_sim

        return {
            'dt_alignment_score_mean': round(float(np.mean(alignment_score)), 4),
            'dt_alignment_score_std': round(float(np.std(alignment_score)), 4),
            'dt_alignment_content_weight': content_weight,
            'dt_alignment_style_weight': style_weight
        }

    def calculate_embedding_diversity(
        self,
        transferred_texts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate diversity of generated texts in style space.
        Measures if the model produces varied outputs or repetitive patterns.

        Args:
            transferred_texts: Transferred texts

        Returns:
            Dictionary with diversity metrics
        """
        print("\n[DeepTransfer] Calculating embedding diversity...")

        # Encode in style space
        style_embs = self.encode_style(transferred_texts)

        # Calculate pairwise distances
        from scipy.spatial.distance import pdist
        pairwise_distances = pdist(style_embs, metric='cosine')

        # Variance in embeddings (higher = more diverse)
        embedding_variance = np.var(style_embs, axis=0).mean()

        return {
            'dt_diversity_pairwise_dist_mean': round(float(np.mean(pairwise_distances)), 4),
            'dt_diversity_pairwise_dist_std': round(float(np.std(pairwise_distances)), 4),
            'dt_diversity_embedding_variance': round(float(embedding_variance), 4)
        }

    def evaluate_all(
        self,
        source_texts: List[str],
        reference_texts: List[str],
        transferred_texts: List[str],
        content_weight: float = 0.5,
        style_weight: float = 0.5
    ) -> Dict[str, float]:
        """
        Run all DeepTransfer-specific metrics.

        Args:
            source_texts: De-styled source texts
            reference_texts: Target style references
            transferred_texts: Transferred texts
            content_weight: Weight for content in alignment score
            style_weight: Weight for style in alignment score

        Returns:
            Dictionary with all DeepTransfer metrics
        """
        results = {}

        # 1. Content Preservation
        results.update(self.calculate_content_preservation(source_texts, transferred_texts))

        # 2. Style Similarity
        results.update(self.calculate_style_similarity(reference_texts, transferred_texts))

        # 3. Content-Style Disentanglement
        results.update(self.calculate_content_style_disentanglement(
            source_texts, reference_texts, transferred_texts
        ))

        # 4. Dual-Space Alignment
        results.update(self.calculate_dual_space_alignment(
            source_texts, reference_texts, transferred_texts,
            content_weight, style_weight
        ))

        # 5. Embedding Diversity
        results.update(self.calculate_embedding_diversity(transferred_texts))

        return results
