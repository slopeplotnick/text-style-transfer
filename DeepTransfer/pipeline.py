import os
import glob
from tqdm import tqdm
from typing import Optional
from config import Config
from data_processing import process_file_to_chunks, split_into_paragraphs, detect_section, process_file_to_paragraphs, split_into_sentences
from encoders import DualEncoder
from vector_store import VectorStore
from llm_client import LLMClient

class DeepTransferPipeline:
    def __init__(self, config=Config, use_journal_filter=None, use_section_filter=None):
        self.config = config

        # Allow runtime override of filter settings
        self.use_journal_filter = use_journal_filter if use_journal_filter is not None else config.USE_JOURNAL_FILTER
        self.use_section_filter = use_section_filter if use_section_filter is not None else config.USE_SECTION_FILTER

        self.encoders = DualEncoder(config.CONTENT_ENCODER_MODEL, config.STYLE_ENCODER_MODEL)
        self.store = VectorStore()
        self.llm = LLMClient(config.API_URL, config.API_KEY, config.MODEL_NAME)

        print(f"DeepTransfer Pipeline initialized:")
        print(f"  - Use journal filter: {self.use_journal_filter}")
        print(f"  - Use section filter: {self.use_section_filter}")

        # Load index if exists
        if os.path.exists(config.INDEX_FILE):
            self.store.load(config.INDEX_FILE)

    def build_index(self, data_dir: str):
        """
        Builds dual-granularity vector index from text files in data_dir (recursively).
        Creates both chunk-level and paragraph-level indices.
        """
        print(f"Building dual-granularity index from {data_dir} (including subdirectories)...")
        txt_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)

        if self.config.USE_DUAL_GRANULARITY:
            # Build both chunk-level and paragraph-level indices
            self._build_chunk_index(txt_files)
            self._build_paragraph_index(txt_files)
        else:
            # Legacy: only chunk-level
            self._build_chunk_index(txt_files)

        self.store.save(self.config.INDEX_FILE)

    def _build_chunk_index(self, txt_files):
        """Build chunk-level index."""
        print("\n[Chunk-level Index]")
        all_chunks = []
        all_metadatas = []

        # Step 1: Process files into chunks
        for fpath in tqdm(txt_files, desc="Processing files (chunks)"):
            file_data = process_file_to_chunks(fpath, self.config.CHUNK_SIZE)

            for item in file_data:
                all_chunks.append(item['chunk_text'])
                all_metadatas.append({
                    "source": item['source'],
                    "paragraph_context": item['paragraph_text'],
                    "section_type": item['section_type'],
                    "journal": item['journal']
                })

        if not all_chunks:
            print("No valid chunks found. Check data directory.")
            return

        # Step 2: Encode chunks
        content_embs = self.encoders.encode_content(all_chunks)
        style_embs = self.encoders.encode_style(all_chunks)

        # Step 3: Store chunk-level data
        self.store.add_chunks(all_chunks, content_embs, style_embs, all_metadatas)
        print(f"Chunk-level index built: {len(all_chunks)} chunks")

    def _build_paragraph_index(self, txt_files):
        """Build paragraph-level index."""
        print("\n[Paragraph-level Index]")
        all_paragraphs = []
        all_metadatas = []

        # Step 1: Process files into paragraphs
        for fpath in tqdm(txt_files, desc="Processing files (paragraphs)"):
            para_data = process_file_to_paragraphs(fpath)

            for item in para_data:
                all_paragraphs.append(item['paragraph_text'])
                all_metadatas.append({
                    "source": item['source'],
                    "section_type": item['section_type'],
                    "journal": item['journal']
                })

        if not all_paragraphs:
            print("No valid paragraphs found. Check data directory.")
            return

        # Step 2: Encode paragraphs
        content_embs = self.encoders.encode_content(all_paragraphs)
        style_embs = self.encoders.encode_style(all_paragraphs)

        # Step 3: Store paragraph-level data
        self.store.add_paragraphs(all_paragraphs, content_embs, style_embs, all_metadatas)
        print(f"Paragraph-level index built: {len(all_paragraphs)} paragraphs")

    def _transfer_paragraph(self, para_text: str, section_type: str, prev_context: str = "", target_journal: Optional[str] = None) -> str:
        """
        Dual-granularity paragraph transfer:
        1. Match paragraph-level templates
        2. Split into chunks and match chunk-level templates
        3. Transform each chunk with LLM
        4. Enhance paragraph coherence using paragraph template
        """
        if not self.config.USE_DUAL_GRANULARITY:
            # Fallback to legacy single-granularity method
            return self._transfer_paragraph_legacy(para_text, section_type, prev_context, target_journal)

        # === Step 1: Paragraph-level Template Matching ===
        para_content_emb = self.encoders.encode_content(para_text)

        # Build filter dict based on configuration
        filter_dict = {}
        if self.use_journal_filter and target_journal:
            filter_dict["journal"] = target_journal
        if self.use_section_filter:
            filter_dict["section_type"] = section_type

        # Retrieve paragraph templates (with optional style-aware reranking)
        if self.config.ENABLE_STYLE_RERANKING:
            para_results = self.store.search_paragraphs_with_reranking(
                para_content_emb,
                top_k=self.config.PARAGRAPH_TOP_K,
                retrieve_k=self.config.STYLE_RERANKING_RETRIEVE_K,
                style_weight=self.config.STYLE_RERANKING_WEIGHT,
                filter_dict=filter_dict
            )
        else:
            para_results = self.store.search_paragraphs(
                para_content_emb,
                top_k=self.config.PARAGRAPH_TOP_K,
                filter_dict=filter_dict
            )

        if not para_results:
            # Fallback: search without filters
            para_results = self.store.search_paragraphs(para_content_emb, top_k=self.config.PARAGRAPH_TOP_K)

        if not para_results:
            return para_text  # Fail safe

        # Use top paragraph template for coherence enhancement
        para_template = para_results[0]['text']
        para_template_metadata = para_results[0]['metadata']

        # === Step 2: Chunk-level Matching and Transformation ===
        sentences = split_into_sentences(para_text)

        # Create chunks from sentences
        chunks = []
        for i in range(0, len(sentences), self.config.CHUNK_SIZE):
            chunk_sents = sentences[i:i + self.config.CHUNK_SIZE]
            chunk_text = " ".join(chunk_sents)
            chunks.append(chunk_text)

        transformed_chunks = []

        for chunk in chunks:
            # Encode chunk
            chunk_emb = self.encoders.encode_content(chunk)

            # Retrieve chunk-level examples (with optional style-aware reranking)
            if self.config.ENABLE_STYLE_RERANKING:
                chunk_results = self.store.search_chunks_with_reranking(
                    chunk_emb,
                    top_k=self.config.CHUNK_TOP_K,
                    retrieve_k=self.config.STYLE_RERANKING_RETRIEVE_K,
                    style_weight=self.config.STYLE_RERANKING_WEIGHT,
                    filter_dict=filter_dict
                )
            else:
                chunk_results = self.store.search_chunks(
                    chunk_emb,
                    top_k=self.config.CHUNK_TOP_K,
                    filter_dict=filter_dict
                )

            if not chunk_results:
                # Fallback: search without filters
                chunk_results = self.store.search_chunks(chunk_emb, top_k=self.config.CHUNK_TOP_K)

            if not chunk_results:
                transformed_chunks.append(chunk)
                continue

            # Transform chunk using LLM
            transformed_chunk = self._transform_chunk_with_templates(
                chunk,
                chunk_results,
                para_template,
                prev_context
            )
            transformed_chunks.append(transformed_chunk)

        # === Step 3: Paragraph-level Coherence Enhancement ===
        if self.config.ENABLE_COHERENCE_ENHANCEMENT and len(transformed_chunks) > 1:
            final_paragraph = self._enhance_paragraph_coherence(
                transformed_chunks,
                para_template,
                target_journal
            )
        else:
            final_paragraph = " ".join(transformed_chunks)

        return final_paragraph

    def _transform_chunk_with_templates(
        self,
        chunk: str,
        chunk_examples: list,
        para_template: str,
        prev_context: str
    ) -> str:
        """
        Transform a chunk using matched chunk-level and paragraph-level templates.
        """
        # Construct vocabulary references
        vocab_refs = "\n".join([f"- {r['text']}" for r in chunk_examples])

        prompt = f"""You are an expert academic editor.

Task: Rewrite the [Input Text] to match the academic style of the [References].

### Context (Preceding Text)
{prev_context[-200:] if prev_context else "(Start of Document)"}

### 1. Vocabulary References (Chunk-level Style)
{vocab_refs}

### 2. Paragraph Structure Reference
*"{para_template[:300]}..."*

### 3. Input Text
{chunk}

### Instructions
- **Tone**: Adopt a formal, objective academic tone.
- **Vocabulary**: Use similar word choices and expressions as the vocabulary references.
- **Coherence**: Ensure smooth continuation from the preceding context.
- **Output**: The rewritten text ONLY (no explanations).
"""
        return self.llm.generate(prompt)

    def _enhance_paragraph_coherence(
        self,
        transformed_chunks: list,
        para_template: str,
        target_journal: Optional[str] = None
    ) -> str:
        """
        Enhance paragraph-level coherence using paragraph template.
        Similar to ZeroStylus Phase 2.3.
        """
        combined_text = " ".join(transformed_chunks)

        journal_instruction = ""
        if target_journal:
            journal_instruction = f"Target Journal Style: **{target_journal}**."

        prompt = f"""You are refining a paragraph to improve its coherence and flow.

{journal_instruction}

### Transformed Sentences
{combined_text}

### Target Paragraph Structure Reference
{para_template}

### Instructions
1. Adjust inter-sentence transitions to improve flow
2. Add or refine discourse markers (however, therefore, moreover, etc.)
3. Ensure referential consistency across sentences
4. Maintain the logical progression shown in the reference paragraph
5. Keep the semantic content of each sentence unchanged
6. Match the writing style of the reference paragraph

### Refined Paragraph (output only the paragraph, no explanations):
"""
        return self.llm.generate(prompt)

    def _transfer_paragraph_legacy(self, para_text: str, section_type: str, prev_context: str = "", target_journal: Optional[str] = None) -> str:
        """
        Legacy single-granularity method (original implementation).
        """
        # 1. Encode Input
        input_content_emb = self.encoders.encode_content(para_text)

        results = []
        strategy_used = "Global"

        # 2. Configurable Retrieval Strategy
        # Build filter dict based on configuration
        filter_dict = {}

        # Try with both filters if enabled
        if self.use_journal_filter and self.use_section_filter and target_journal:
            filter_dict = {"journal": target_journal, "section_type": section_type}
            if self.config.ENABLE_STYLE_RERANKING:
                results = self.store.search_chunks_with_reranking(
                    input_content_emb, top_k=2,
                    retrieve_k=self.config.STYLE_RERANKING_RETRIEVE_K,
                    style_weight=self.config.STYLE_RERANKING_WEIGHT,
                    filter_dict=filter_dict
                )
            else:
                results = self.store.search_content(input_content_emb, top_k=2, filter_dict=filter_dict)
            if results:
                strategy_used = f"{target_journal} + {section_type}"

        # Try with journal filter only if enabled
        if not results and self.use_journal_filter and target_journal:
            filter_dict = {"journal": target_journal}
            if self.config.ENABLE_STYLE_RERANKING:
                results = self.store.search_chunks_with_reranking(
                    input_content_emb, top_k=2,
                    retrieve_k=self.config.STYLE_RERANKING_RETRIEVE_K,
                    style_weight=self.config.STYLE_RERANKING_WEIGHT,
                    filter_dict=filter_dict
                )
            else:
                results = self.store.search_content(input_content_emb, top_k=2, filter_dict=filter_dict)
            if results:
                strategy_used = f"{target_journal} (Any Section)"

        # Try with section filter only if enabled
        if not results and self.use_section_filter:
            filter_dict = {"section_type": section_type}
            if self.config.ENABLE_STYLE_RERANKING:
                results = self.store.search_chunks_with_reranking(
                    input_content_emb, top_k=2,
                    retrieve_k=self.config.STYLE_RERANKING_RETRIEVE_K,
                    style_weight=self.config.STYLE_RERANKING_WEIGHT,
                    filter_dict=filter_dict
                )
            else:
                results = self.store.search_content(input_content_emb, top_k=2, filter_dict=filter_dict)
            if results:
                strategy_used = f"{section_type} (Any Journal)"

        # Global fallback (no filters)
        if not results:
            if self.config.ENABLE_STYLE_RERANKING:
                results = self.store.search_chunks_with_reranking(
                    input_content_emb, top_k=2,
                    retrieve_k=self.config.STYLE_RERANKING_RETRIEVE_K,
                    style_weight=self.config.STYLE_RERANKING_WEIGHT
                )
            else:
                results = self.store.search_content(input_content_emb, top_k=2)
            strategy_used = "Global (No Filters)"

        if not results:
            return para_text # Fail safe

        # 3. Construct Prompt
        primary_ref = results[0]
        structural_guide = primary_ref['metadata'].get('paragraph_context', primary_ref['text'])
        vocab_refs = "\n".join([f"- {r['text']}" for r in results])

        # Add specific instruction if specific journal was requested
        journal_instruction = ""
        if target_journal and self.use_journal_filter:
            journal_instruction = f"Target Journal Style: **{target_journal}**."

        prompt = f"""You are an expert academic editor.

Task: Rewrite the [Input Paragraph] to match the academic style of the [References].
{journal_instruction}
Retrieval Strategy Used: {strategy_used}

### Context (Preceding Paragraph)
{prev_context}

### 1. Structural Guide (Target Style)
*"{structural_guide}"*

### 2. Vocabulary References
{vocab_refs}

### 3. Input Paragraph (Draft)
{para_text}

### Instructions
- **Tone**: Adopt a formal, objective academic tone.
- **Flow**: Mimic the structure of the [Structural Guide].
- **Continuity**: Ensure smooth transition from the [Context].
- **Output**: The rewritten paragraph ONLY.
"""
        # 4. Generate
        return self.llm.generate(prompt)

    def transfer_document(self, full_text: str, target_journal: Optional[str] = None) -> str:
        """
        Main entry point for full document transfer.
        """
        paragraphs = split_into_paragraphs(full_text)
        print(f"Document split into {len(paragraphs)} paragraphs.")
        if target_journal:
            print(f"Target Journal: {target_journal}")
        
        final_doc = []
        prev_context = "(Start of Document)"
        current_section = "general"
        
        for i, para in enumerate(tqdm(paragraphs, desc="Transferring Document")):
            # Update section context
            current_section = detect_section(para, current_section)
            
            # Skip very short headers
            if len(para.split()) < 5:
                final_doc.append(para)
                continue
            
            # Transfer with journal constraint
            transferred_para = self._transfer_paragraph(para, current_section, prev_context, target_journal)
            
            # Update context
            prev_context = transferred_para[-200:] 
            
            final_doc.append(transferred_para)
            
        return "\n\n".join(final_doc)

    def transfer_style(self, input_text: str, target_journal: Optional[str] = None) -> str:
        """
        Wrapper for single-text usage.
        """
        return self.transfer_document(input_text, target_journal)