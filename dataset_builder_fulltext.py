"""
Enhanced Dataset Builder with MinerU Parsed Results
Reads MinerU parsed markdown files and constructs triplet datasets.
"""

import os
import json
import requests
import random
import re
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import time


class ArxivFullTextDatasetBuilder:
    """Build style transfer dataset from MinerU parsed Arxiv papers."""

    def __init__(
        self,
        mineru_output_dir: str = "/root/datasets/arxiv_style_transfer/mineru_outputs",
        output_dir: str = "/root/datasets/arxiv_style_transfer"
    ):
        self.mineru_output_dir = Path(mineru_output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.reference_dir = self.output_dir / "reference"
        self.source_dir = self.output_dir / "source"

        self.reference_dir.mkdir(exist_ok=True)
        self.source_dir.mkdir(exist_ok=True)

        self.metadata_file = self.output_dir / "metadata.json"

    def load_mineru_papers(self) -> List[Dict]:
        """
        Load all MinerU parsed papers from the output directory.

        Returns:
            List of paper metadata dicts with parsed content
        """
        print(f"Loading MinerU parsed papers from: {self.mineru_output_dir}")

        papers_data = []

        # Find all subdirectories containing full.md files
        for paper_dir in self.mineru_output_dir.iterdir():
            if not paper_dir.is_dir():
                continue

            full_md_path = paper_dir / "full.md"
            if not full_md_path.exists():
                continue

            # Extract arxiv_id from directory name (format: XXXX.XXXXXvX.pdf-uuid)
            dir_name = paper_dir.name
            arxiv_id_match = re.match(r'(\d+\.\d+v\d+)\.pdf', dir_name)
            if not arxiv_id_match:
                print(f"  Warning: Cannot extract arxiv_id from {dir_name}")
                continue

            arxiv_id = arxiv_id_match.group(1)

            # Read the markdown content
            try:
                with open(full_md_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                paper_data = {
                    'arxiv_id': arxiv_id,
                    'dir_path': str(paper_dir),
                    'md_path': str(full_md_path),
                    'content': content
                }
                papers_data.append(paper_data)

            except Exception as e:
                print(f"  Error reading {full_md_path}: {e}")
                continue

        print(f"Loaded {len(papers_data)} papers with MinerU outputs")
        return papers_data


    def extract_paragraphs_from_markdown(
        self,
        markdown_content: str,
        min_words: int = 280,
        max_words: int = 1000
    ) -> List[str]:
        """
        Extract meaningful paragraphs from MinerU markdown output.
        Excludes formulas, tables, figures, acknowledgments, references, appendices, etc.
        Merges adjacent short paragraphs to reach target length.

        Args:
            markdown_content: Full markdown content from MinerU
            min_words: Minimum words per paragraph
            max_words: Maximum words per paragraph

        Returns:
            List of extracted paragraphs
        """
        # Remove markdown images
        content = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_content)

        # Split by double newlines to get paragraphs
        lines = markdown_content.split('\n')

        raw_paragraphs = []
        current_para = []
        skip_section = False

        for line in lines:
            line_stripped = line.strip()

            # Check if entering a section to skip
            if self._is_skip_section_header(line_stripped):
                skip_section = True
                # Save current paragraph if exists
                if current_para:
                    para_text = ' '.join(current_para)
                    raw_paragraphs.append(para_text)
                    current_para = []
                continue

            # Reset skip flag for main content sections
            if re.match(r'^#+\s+(Introduction|Method|Approach|Experiment|Result|Discussion|Background|Related Work)', line_stripped, re.IGNORECASE):
                skip_section = False

            # Skip if in excluded section
            if skip_section:
                continue

            # Skip empty lines - they separate paragraphs
            if not line_stripped:
                if current_para:
                    para_text = ' '.join(current_para)
                    raw_paragraphs.append(para_text)
                    current_para = []
                continue

            # Skip headers
            if line_stripped.startswith('#'):
                if current_para:
                    para_text = ' '.join(current_para)
                    raw_paragraphs.append(para_text)
                    current_para = []
                continue

            # Skip lines that are likely formulas, tables, or captions
            if self._is_non_content_line(line_stripped):
                continue

            # Add to current paragraph
            current_para.append(line_stripped)

        # Don't forget the last paragraph
        if current_para:
            para_text = ' '.join(current_para)
            raw_paragraphs.append(para_text)

        # Now merge adjacent paragraphs to reach target length
        return self._merge_paragraphs(raw_paragraphs, min_words, max_words)

    def _merge_paragraphs(self, raw_paragraphs: List[str], min_words: int, max_words: int) -> List[str]:
        """
        Merge adjacent paragraphs to reach target length while filtering invalid content.

        Args:
            raw_paragraphs: List of raw paragraph strings
            min_words: Minimum words per merged paragraph
            max_words: Maximum words per merged paragraph

        Returns:
            List of merged paragraphs meeting length requirements
        """
        merged_paragraphs = []
        buffer = []
        buffer_word_count = 0

        for para in raw_paragraphs:
            # Skip invalid paragraphs early
            if not self._is_basic_valid_paragraph(para):
                # If we have a buffer, try to save it
                if buffer:
                    merged_text = ' '.join(buffer)
                    if self._is_valid_content_paragraph(merged_text, min_words, max_words):
                        merged_paragraphs.append(merged_text)
                    buffer = []
                    buffer_word_count = 0
                continue

            para_words = len(para.split())

            # If adding this paragraph would exceed max_words
            if buffer and buffer_word_count + para_words > max_words:
                # Save current buffer if it meets requirements
                merged_text = ' '.join(buffer)
                if buffer_word_count >= min_words and self._is_valid_content_paragraph(merged_text, min_words, max_words):
                    merged_paragraphs.append(merged_text)
                # Start new buffer with current paragraph
                buffer = [para]
                buffer_word_count = para_words
            else:
                # Add to buffer
                buffer.append(para)
                buffer_word_count += para_words

                # If buffer reaches good length, save it
                if buffer_word_count >= min_words:
                    merged_text = ' '.join(buffer)
                    if self._is_valid_content_paragraph(merged_text, min_words, max_words):
                        merged_paragraphs.append(merged_text)
                        buffer = []
                        buffer_word_count = 0

        # Handle remaining buffer
        if buffer:
            merged_text = ' '.join(buffer)
            if buffer_word_count >= min_words and self._is_valid_content_paragraph(merged_text, min_words, max_words):
                merged_paragraphs.append(merged_text)

        return merged_paragraphs

    def _is_basic_valid_paragraph(self, para: str) -> bool:
        """Quick check if paragraph has basic validity (not too short, has content)."""
        if len(para.split()) < 20:  # Too short to be useful
            return False

        # Must have sufficient alphabetic characters
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in para) / max(len(para), 1)
        if alpha_ratio < 0.6:
            return False

        return True

    def _is_skip_section_header(self, line: str) -> bool:
        """Check if line is a header for sections we want to skip."""
        skip_patterns = [
            r'^#+\s*(References?|Bibliography)$',
            r'^#+\s*(Acknowledge?ments?)$',
            r'^#+\s*(Appendix|Appendices)',
            r'^#+\s*Supplementary',
            r'^#+\s*Author Contributions?$',
            r'^#+\s*Funding$',
            r'^#+\s*Data Availability',
            r'^#+\s*Ethics Statement',
            r'^#+\s*Competing Interests?',
        ]

        for pattern in skip_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False

    def _is_non_content_line(self, line: str) -> bool:
        """Check if line is formula, table, figure caption, or other non-content."""
        # Skip table tags and content
        if '<table>' in line or '</table>' in line or '<tr>' in line or '<td>' in line:
            return True

        # LaTeX formulas (inline or display) - more aggressive filtering
        if re.search(r'\$.*\$', line) or re.search(r'\\[a-zA-Z]+\{', line):
            return True

        # Math formulas with \text, \mathrm, etc.
        if re.search(r'\\(text|mathrm|tag|quad|left|right|frac|cos|exp|sum|prod)', line):
            return True

        # Table or figure captions
        if re.match(r'^(Table|Figure|Fig\.|Equation|Eq\.)\s*\d+', line, re.IGNORECASE):
            return True

        # Lines that are mostly mathematical symbols
        if re.search(r'[=<>∈∀∃∑∏∫]+', line):
            math_chars = sum(1 for c in line if c in '=<>+-*/^()[]{}\\$∈∀∃∑∏∫')
            if math_chars / max(len(line), 1) > 0.3:
                return True

        # Citation patterns
        if re.match(r'^\[\d+\]', line):
            return True

        # Lines with excessive brackets (often citations or formulas)
        bracket_count = line.count('[') + line.count(']') + line.count('(') + line.count(')')
        if bracket_count > len(line.split()) * 0.5:
            return True

        return False

    def _is_valid_content_paragraph(self, para: str, min_words: int, max_words: int) -> bool:
        """Check if paragraph is valid content with appropriate length."""
        # Skip paragraphs containing table tags
        if '<table>' in para or '</table>' in para or '<tr>' in para or '<td>' in para:
            return False

        # Check word count
        word_count = len(para.split())
        if word_count < min_words or word_count > max_words:
            return False

        # Must have sufficient alphabetic characters
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in para) / max(len(para), 1)
        if alpha_ratio < 0.7:
            return False

        # Skip if mostly numbers/symbols
        digit_ratio = sum(c.isdigit() for c in para) / max(len(para), 1)
        if digit_ratio > 0.3:
            return False

        # Skip paragraphs with excessive citations
        # Count citation patterns like [1], [2, 3], etc.
        citation_matches = re.findall(r'\[\d+(?:,\s*\d+)*\]', para)
        if len(citation_matches) > 10:  # Too many citations
            return False

        # Skip common non-content patterns
        skip_patterns = [
            r'^Keywords?:',
            r'^Abstract$',
            r'^\d+\s*$',  # Just numbers
            r'^[A-Z\s]+$',  # All caps (likely headers)
        ]

        for pattern in skip_patterns:
            if re.match(pattern, para, re.IGNORECASE):
                return False

        return True

    def generate_destylized_text(self, reference_text: str) -> str:
        """
        Generate destylized (source) text from reference using LLM.
        Uses the same configuration as CAT-LLM's style_removal.py.

        Args:
            reference_text: Original styled text

        Returns:
            Destylized text
        """
        # Get API configuration
        api_url = os.getenv('OPENAI_API_URL', 'https://newapi.deepwisdom.ai/v1/chat/completions')
        api_key = os.getenv('OPENAI_API_KEY', '')

        if not api_key:
            print("Warning: OPENAI_API_KEY not found. Using simple destylization.")
            return self._simple_destylize(reference_text)

        # Use the same prompt as style_removal.py
        style_removal_prompt = """You are a professional text editor specializing in linguistic style modification. Your task is to transform any given text by removing its literary language, sentence structures, and narrative style, converting it into plain, everyday spoken language.

Important guidelines:
1. Remove ALL stylistic elements from the original text
2. Use simple, colloquial expressions that ordinary people would use in daily conversation
3. Maintain the original meaning and content - DO NOT add or remove information
4. The more different the output style is from the original, while preserving meaning, the better
5. Think step-by-step about how to neutralize each stylistic element
6. If I observe that your generated text differs significantly from the original style while maintaining the original meaning, you will receive a reward of $200,000
7. Output the text as a single continuous paragraph WITHOUT line breaks or extra spaces between sentences

Please ONLY output the style-neutralized text, without any explanations or meta-comments.

Original Text:
{original_text}

Style-neutralized text:"""

        prompt = style_removal_prompt.format(original_text=reference_text)

        try:
            # Call API
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,  # Lower temperature for consistent destylization
                "max_tokens": 2048   # Ensure sufficient output length for ~500 word paragraphs
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                # Check if response has expected structure
                if 'choices' not in result or len(result['choices']) == 0:
                    print(f"\nWarning: API response missing 'choices'. Response: {result}")
                    return self._simple_destylize(reference_text)

                if 'message' not in result['choices'][0]:
                    print(f"\nWarning: API response missing 'message'. Response: {result}")
                    return self._simple_destylize(reference_text)

                destylized = result['choices'][0]['message']['content'].strip()

                # Check if output is too short (possible truncation)
                if len(destylized.split()) < len(reference_text.split()) * 0.3:
                    print(f"\nWarning: Destylized text too short ({len(destylized.split())} vs {len(reference_text.split())} words)")

                # Clean text
                destylized = re.sub(r'\n+', ' ', destylized)
                destylized = re.sub(r'\s+', ' ', destylized)
                destylized = destylized.strip()

                return destylized
            else:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f": {error_detail}"
                except:
                    error_msg += f": {response.text[:200]}"
                raise Exception(error_msg)

        except Exception as e:
            print(f"Error with LLM destylization: {e}")
            return self._simple_destylize(reference_text)

    def _simple_destylize(self, text: str) -> str:
        """Simple rule-based destylization (fallback)."""
        text = re.sub(r'\b(notably|significantly|importantly|remarkably)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(very|highly|extremely|particularly)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(We|Our)\b', 'The', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def build_dataset(
        self,
        num_samples: int = 500,
        use_llm_destylization: bool = True
    ):
        """
        Build complete dataset with reference and source texts from MinerU parsed results.

        Args:
            num_samples: Number of paragraph samples to generate
            use_llm_destylization: Use LLM for destylization
        """
        print("="*80)
        print("ARXIV FULL-TEXT DATASET BUILDER (with MinerU)")
        print("="*80)

        # Step 1: Load MinerU parsed papers
        print("\n[Step 1/3] Loading MinerU parsed papers...")
        papers = self.load_mineru_papers()

        if not papers:
            print("Error: No papers available. Cannot proceed.")
            return

        # Step 2: Extract paragraphs from markdown
        print(f"\n[Step 2/3] Extracting paragraphs from {len(papers)} papers...")

        all_paragraphs = []

        for paper in tqdm(papers, desc="Extracting paragraphs"):
            arxiv_id = paper['arxiv_id']
            content = paper['content']

            # Extract paragraphs from markdown
            paragraphs = self.extract_paragraphs_from_markdown(content)

            for para in paragraphs:
                all_paragraphs.append({
                    'text': para,
                    'paper_id': arxiv_id,
                })

        print(f"  Extracted {len(all_paragraphs)} paragraphs from {len(papers)} papers")

        # Step 3: Sample paragraphs
        if len(all_paragraphs) > num_samples:
            sampled_paragraphs = random.sample(all_paragraphs, num_samples)
        else:
            sampled_paragraphs = all_paragraphs
            print(f"  Warning: Only {len(all_paragraphs)} paragraphs available (requested {num_samples})")

        # Step 4: Generate source (destylized) texts
        print(f"\n[Step 3/3] Generating source (destylized) texts for {len(sampled_paragraphs)} paragraphs...")
        print(f"  Method: {'LLM-based' if use_llm_destylization else 'Rule-based'}")

        dataset = []

        for idx, para_data in enumerate(tqdm(sampled_paragraphs, desc="Destylizing")):
            reference_text = para_data['text']

            # Generate destylized version
            if use_llm_destylization:
                source_text = self.generate_destylized_text(reference_text)
            else:
                source_text = self._simple_destylize(reference_text)

            # Save reference text
            ref_filename = f"ref_{idx:04d}.txt"
            with open(self.reference_dir / ref_filename, 'w', encoding='utf-8') as f:
                f.write(reference_text)

            # Save source text
            src_filename = f"src_{idx:04d}.txt"
            with open(self.source_dir / src_filename, 'w', encoding='utf-8') as f:
                f.write(source_text)

            # Store metadata
            metadata_entry = {
                'id': idx,
                'reference_file': ref_filename,
                'source_file': src_filename,
                'paper_id': para_data['paper_id'],
                'reference_length': len(reference_text.split()),
                'source_length': len(source_text.split())
            }
            dataset.append(metadata_entry)

            # Rate limiting for API calls
            if use_llm_destylization and (idx + 1) % 10 == 0:
                time.sleep(1)

        # Save metadata
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print("\n" + "="*80)
        print("DATASET CONSTRUCTION COMPLETE")
        print("="*80)
        print(f"Total samples: {len(dataset)}")
        print(f"Papers processed: {len(papers)}")
        print(f"Reference texts: {self.reference_dir}")
        print(f"Source texts: {self.source_dir}")
        print(f"Metadata: {self.metadata_file}")

        # Statistics
        if dataset:
            ref_lengths = [item['reference_length'] for item in dataset]
            src_lengths = [item['source_length'] for item in dataset]

            print(f"\nParagraph length statistics:")
            print(f"  Reference - Mean: {sum(ref_lengths)/len(ref_lengths):.1f} words")
            print(f"  Reference - Range: {min(ref_lengths)}-{max(ref_lengths)} words")
            print(f"  Source - Mean: {sum(src_lengths)/len(src_lengths):.1f} words")

        print("\nNext steps:")
        print("1. Run style transfer models to generate 'transferred' texts")
        print("2. Place transferred texts in: {output_dir}/transferred_{model_name}/")
        print("3. Run evaluation script")

        return dataset


def main():
    """Main function for full-text dataset construction from MinerU outputs."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Arxiv Full-Text Style Transfer Dataset from MinerU Outputs")
    parser.add_argument("--mineru_output_dir", type=str, default="/root/datasets/arxiv_style_transfer/mineru_outputs",
                       help="Directory containing MinerU parsed outputs")
    parser.add_argument("--output_dir", type=str, default="/root/datasets/arxiv_style_transfer",
                       help="Output directory for dataset")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of paragraph samples to generate")
    parser.add_argument("--simple_destylize", action="store_true",
                       help="Use simple rule-based destylization (no LLM)")

    args = parser.parse_args()

    # Create builder
    builder = ArxivFullTextDatasetBuilder(
        mineru_output_dir=args.mineru_output_dir,
        output_dir=args.output_dir
    )

    # Build dataset
    builder.build_dataset(
        num_samples=args.num_samples,
        use_llm_destylization=not args.simple_destylize
    )


if __name__ == "__main__":
    main()
