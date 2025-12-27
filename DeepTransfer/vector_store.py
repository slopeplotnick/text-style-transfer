import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

class VectorStore:
    """
    Dual-granularity in-memory vector store.
    Supports both chunk-level and paragraph-level indexing and retrieval.
    """

    def __init__(self):
        # Chunk-level storage
        self.chunk_documents: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.chunk_content_embeddings: np.ndarray = None
        self.chunk_style_embeddings: np.ndarray = None

        # Paragraph-level storage
        self.para_documents: List[str] = []
        self.para_metadata: List[Dict[str, Any]] = []
        self.para_content_embeddings: np.ndarray = None
        self.para_style_embeddings: np.ndarray = None

    def add_chunks(self, texts: List[str], content_embs: np.ndarray, style_embs: np.ndarray, metadatas: List[Dict] = None):
        """
        Add chunk-level documents and their embeddings to the store.
        """
        self.chunk_documents.extend(texts)
        if metadatas:
            self.chunk_metadata.extend(metadatas)
        else:
            self.chunk_metadata.extend([{} for _ in texts])

        if self.chunk_content_embeddings is None:
            self.chunk_content_embeddings = content_embs
            self.chunk_style_embeddings = style_embs
        else:
            self.chunk_content_embeddings = np.vstack([self.chunk_content_embeddings, content_embs])
            self.chunk_style_embeddings = np.vstack([self.chunk_style_embeddings, style_embs])

    def add_paragraphs(self, texts: List[str], content_embs: np.ndarray, style_embs: np.ndarray, metadatas: List[Dict] = None):
        """
        Add paragraph-level documents and their embeddings to the store.
        """
        self.para_documents.extend(texts)
        if metadatas:
            self.para_metadata.extend(metadatas)
        else:
            self.para_metadata.extend([{} for _ in texts])

        if self.para_content_embeddings is None:
            self.para_content_embeddings = content_embs
            self.para_style_embeddings = style_embs
        else:
            self.para_content_embeddings = np.vstack([self.para_content_embeddings, content_embs])
            self.para_style_embeddings = np.vstack([self.para_style_embeddings, style_embs])

    def add_documents(self, texts: List[str], content_embs: np.ndarray, style_embs: np.ndarray, metadatas: List[Dict] = None):
        """
        Deprecated: Use add_chunks or add_paragraphs instead.
        For backward compatibility, adds to chunk-level storage.
        """
        self.add_chunks(texts, content_embs, style_embs, metadatas)

    def search_chunks(self, query_emb: np.ndarray, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve chunk-level documents based on content similarity.
        """
        if self.chunk_content_embeddings is None:
            return []

        # query_emb shape: (1, dim) or (dim,)
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)

        # 1. Calculate Cosine similarity for ALL documents first (fast vectorized op)
        sims = cosine_similarity(query_emb, self.chunk_content_embeddings)[0]

        # 2. Apply Filtering
        valid_indices = []
        for i in range(len(self.chunk_documents)):
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    if self.chunk_metadata[i].get(k) != v:
                        match = False
                        break
                if match:
                    valid_indices.append(i)
            else:
                valid_indices.append(i)

        # If filtering removed everything, fallback to no filter
        if not valid_indices and filter_dict:
             valid_indices = range(len(self.chunk_documents))

        # 3. Sort and Retrieve from valid indices
        valid_scores = sims[valid_indices]
        sorted_args = np.argsort(valid_scores)[::-1][:top_k]

        results = []
        for rank_idx in sorted_args:
            original_idx = valid_indices[rank_idx]
            results.append({
                "text": self.chunk_documents[original_idx],
                "score": float(sims[original_idx]),
                "metadata": self.chunk_metadata[original_idx],
                "style_embedding": self.chunk_style_embeddings[original_idx]
            })

        return results

    def search_paragraphs(self, query_emb: np.ndarray, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve paragraph-level documents based on content similarity.
        """
        if self.para_content_embeddings is None:
            return []

        # query_emb shape: (1, dim) or (dim,)
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)

        # 1. Calculate Cosine similarity for ALL documents first
        sims = cosine_similarity(query_emb, self.para_content_embeddings)[0]

        # 2. Apply Filtering
        valid_indices = []
        for i in range(len(self.para_documents)):
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    if self.para_metadata[i].get(k) != v:
                        match = False
                        break
                if match:
                    valid_indices.append(i)
            else:
                valid_indices.append(i)

        # If filtering removed everything, fallback to no filter
        if not valid_indices and filter_dict:
             valid_indices = range(len(self.para_documents))

        # 3. Sort and Retrieve from valid indices
        valid_scores = sims[valid_indices]
        sorted_args = np.argsort(valid_scores)[::-1][:top_k]

        results = []
        for rank_idx in sorted_args:
            original_idx = valid_indices[rank_idx]
            results.append({
                "text": self.para_documents[original_idx],
                "score": float(sims[original_idx]),
                "metadata": self.para_metadata[original_idx],
                "style_embedding": self.para_style_embeddings[original_idx]
            })

        return results

    def search_chunks_with_reranking(
        self,
        query_emb: np.ndarray,
        top_k: int = 3,
        retrieve_k: int = 10,
        style_weight: float = 0.3,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve chunk-level documents with style-aware reranking.

        Strategy:
        1. Retrieve top retrieve_k candidates based on content similarity
        2. Use the top-1 result's style embedding as the target style reference
        3. Rerank all candidates based on their style similarity to the reference
        4. Return top_k results after reranking

        Args:
            query_emb: Query content embedding
            top_k: Number of final results to return
            retrieve_k: Number of candidates to retrieve before reranking
            style_weight: Weight for style similarity (0.0-1.0)
                         Final score = content_score * (1 - style_weight) + style_score * style_weight
            filter_dict: Metadata filters

        Returns:
            List of reranked results with updated scores
        """
        if self.chunk_content_embeddings is None or self.chunk_style_embeddings is None:
            return []

        # Step 1: Retrieve more candidates based on content similarity
        candidates = self.search_chunks(query_emb, top_k=retrieve_k, filter_dict=filter_dict)

        if not candidates:
            return []

        if len(candidates) <= 1 or style_weight == 0.0:
            # No reranking needed
            return candidates[:top_k]

        # Step 2: Use top-1 result's style as reference
        reference_style_emb = candidates[0]['style_embedding'].reshape(1, -1)

        # Step 3: Calculate style similarity for all candidates
        for candidate in candidates:
            candidate_style_emb = candidate['style_embedding'].reshape(1, -1)
            style_sim = cosine_similarity(reference_style_emb, candidate_style_emb)[0][0]

            # Combine content and style scores
            content_score = candidate['score']
            reranked_score = content_score * (1 - style_weight) + style_sim * style_weight

            candidate['style_similarity'] = float(style_sim)
            candidate['content_score'] = content_score
            candidate['reranked_score'] = float(reranked_score)

        # Step 4: Sort by reranked score
        reranked_candidates = sorted(candidates, key=lambda x: x['reranked_score'], reverse=True)

        # Update 'score' field to reflect reranked score
        for candidate in reranked_candidates:
            candidate['score'] = candidate['reranked_score']

        return reranked_candidates[:top_k]

    def search_paragraphs_with_reranking(
        self,
        query_emb: np.ndarray,
        top_k: int = 3,
        retrieve_k: int = 10,
        style_weight: float = 0.3,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve paragraph-level documents with style-aware reranking.

        Strategy:
        1. Retrieve top retrieve_k candidates based on content similarity
        2. Use the top-1 result's style embedding as the target style reference
        3. Rerank all candidates based on their style similarity to the reference
        4. Return top_k results after reranking

        Args:
            query_emb: Query content embedding
            top_k: Number of final results to return
            retrieve_k: Number of candidates to retrieve before reranking
            style_weight: Weight for style similarity (0.0-1.0)
            filter_dict: Metadata filters

        Returns:
            List of reranked results with updated scores
        """
        if self.para_content_embeddings is None or self.para_style_embeddings is None:
            return []

        # Step 1: Retrieve more candidates based on content similarity
        candidates = self.search_paragraphs(query_emb, top_k=retrieve_k, filter_dict=filter_dict)

        if not candidates:
            return []

        if len(candidates) <= 1 or style_weight == 0.0:
            # No reranking needed
            return candidates[:top_k]

        # Step 2: Use top-1 result's style as reference
        reference_style_emb = candidates[0]['style_embedding'].reshape(1, -1)

        # Step 3: Calculate style similarity for all candidates
        for candidate in candidates:
            candidate_style_emb = candidate['style_embedding'].reshape(1, -1)
            style_sim = cosine_similarity(reference_style_emb, candidate_style_emb)[0][0]

            # Combine content and style scores
            content_score = candidate['score']
            reranked_score = content_score * (1 - style_weight) + style_sim * style_weight

            candidate['style_similarity'] = float(style_sim)
            candidate['content_score'] = content_score
            candidate['reranked_score'] = float(reranked_score)

        # Step 4: Sort by reranked score
        reranked_candidates = sorted(candidates, key=lambda x: x['reranked_score'], reverse=True)

        # Update 'score' field to reflect reranked score
        for candidate in reranked_candidates:
            candidate['score'] = candidate['reranked_score']

        return reranked_candidates[:top_k]

    def search_content(self, query_emb: np.ndarray, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Deprecated: Use search_chunks or search_paragraphs instead.
        For backward compatibility, searches chunk-level storage.
        """
        return self.search_chunks(query_emb, top_k, filter_dict)

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump({
                # Chunk-level data
                "chunk_documents": self.chunk_documents,
                "chunk_metadata": self.chunk_metadata,
                "chunk_content_embeddings": self.chunk_content_embeddings,
                "chunk_style_embeddings": self.chunk_style_embeddings,
                # Paragraph-level data
                "para_documents": self.para_documents,
                "para_metadata": self.para_metadata,
                "para_content_embeddings": self.para_content_embeddings,
                "para_style_embeddings": self.para_style_embeddings
            }, f)
        print(f"Index saved to {file_path}")
        print(f"  - Chunks: {len(self.chunk_documents)}")
        print(f"  - Paragraphs: {len(self.para_documents)}")

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"Index file {file_path} not found.")
            return

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

            # Load chunk-level data
            self.chunk_documents = data.get("chunk_documents", data.get("documents", []))
            self.chunk_metadata = data.get("chunk_metadata", data.get("metadata", []))
            self.chunk_content_embeddings = data.get("chunk_content_embeddings", data.get("content_embeddings"))
            self.chunk_style_embeddings = data.get("chunk_style_embeddings", data.get("style_embeddings"))

            # Load paragraph-level data
            self.para_documents = data.get("para_documents", [])
            self.para_metadata = data.get("para_metadata", [])
            self.para_content_embeddings = data.get("para_content_embeddings")
            self.para_style_embeddings = data.get("para_style_embeddings")

        print(f"Index loaded from {file_path}")
        print(f"  - Chunks: {len(self.chunk_documents)}")
        print(f"  - Paragraphs: {len(self.para_documents)}")
