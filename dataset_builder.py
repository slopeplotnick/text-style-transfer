"""
Dataset Builder for Style Transfer Evaluation
Downloads and constructs triplet datasets (reference, source, transferred) from Arxiv papers.

Based on:
- ArxivPapers dataset (Kardas et al. 2020)
- Arxiv-10 dataset (Farhangi et al. 2022)
"""

import os
import json
import requests
import arxiv
import random
import re
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import time


class ArxivDatasetBuilder:
    """Build style transfer dataset from Arxiv papers."""

    def __init__(self, output_dir: str = "/root/datasets/arxiv_style_transfer"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.reference_dir = self.output_dir / "reference"
        self.source_dir = self.output_dir / "source"
        self.reference_dir.mkdir(exist_ok=True)
        self.source_dir.mkdir(exist_ok=True)

        self.metadata_file = self.output_dir / "metadata.json"
        self.metadata = []

    def download_arxiv_papers(
        self,
        categories: List[str] = ["cs.CL", "cs.AI", "cs.LG"],
        num_papers: int = 600,
        max_results_per_category: int = 300
    ):
        """
        Download papers from Arxiv API.

        Args:
            categories: List of arxiv categories
            num_papers: Total number of papers to download
            max_results_per_category: Max papers per category
        """
        print(f"Downloading papers from Arxiv...")
        print(f"Categories: {categories}")

        client = arxiv.Client()
        papers_data = []

        for category in categories:
            print(f"\nFetching from category: {category}")

            # Calculate how many papers needed from this category
            remaining = num_papers - len(papers_data)
            if remaining <= 0:
                break

            fetch_count = min(max_results_per_category, remaining)

            # Search query
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=fetch_count,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            try:
                results = list(client.results(search))
                print(f"  Found {len(results)} papers")

                for paper in results:
                    paper_data = {
                        'arxiv_id': paper.entry_id.split('/')[-1],
                        'title': paper.title,
                        'authors': [author.name for author in paper.authors],
                        'abstract': paper.summary,
                        'categories': paper.categories,
                        'published': str(paper.published),
                        'pdf_url': paper.pdf_url
                    }
                    papers_data.append(paper_data)

                    if len(papers_data) >= num_papers:
                        break

            except Exception as e:
                print(f"  Error fetching from {category}: {e}")
                continue

            if len(papers_data) >= num_papers:
                break

            # Rate limiting
            time.sleep(1)

        print(f"\nTotal papers collected: {len(papers_data)}")

        # Save papers metadata
        with open(self.output_dir / "papers_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False)

        return papers_data

    def extract_paragraphs_from_abstract(self, abstract: str, min_words: int = 30) -> List[str]:
        """
        Extract paragraphs from abstract.

        Args:
            abstract: Paper abstract text
            min_words: Minimum words per paragraph

        Returns:
            List of paragraph strings
        """
        # Clean abstract
        abstract = re.sub(r'\s+', ' ', abstract).strip()

        # Split by newlines or multiple spaces
        paragraphs = re.split(r'\n\n+|\n(?=[A-Z])', abstract)

        # Filter short paragraphs
        valid_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            word_count = len(para.split())
            if word_count >= min_words:
                valid_paragraphs.append(para)

        # If abstract is short, treat whole abstract as one paragraph
        if not valid_paragraphs and len(abstract.split()) >= min_words:
            valid_paragraphs = [abstract]

        return valid_paragraphs

    def generate_destylized_text(self, reference_text: str) -> str:
        """
        Generate destylized (source) text from reference using LLM.
        This removes stylistic features while preserving content.
        Uses the same prompt and API as style_removal.py from CAT-LLM.

        Args:
            reference_text: Original styled text

        Returns:
            Destylized text
        """
        # Get API configuration (same as style_removal.py)
        api_url = os.getenv('OPENAI_API_URL', 'https://yunwu.zeabur.app/v1/chat/completions')
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
            # Call API (using lower temperature for consistency in dataset construction)
            payload = {
                "model": "gpt-5-2025-08-07",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3  # Lower temperature for consistent destylization
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
                destylized = result['choices'][0]['message']['content'].strip()

                # Clean text (remove extra newlines, same as style_removal.py)
                destylized = re.sub(r'\n+', ' ', destylized)
                destylized = re.sub(r'\s+', ' ', destylized)
                destylized = destylized.strip()

                return destylized
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error with LLM destylization: {e}")
            return self._simple_destylize(reference_text)

    def _simple_destylize(self, text: str) -> str:
        """
        Simple rule-based destylization (fallback method).

        Args:
            text: Original text

        Returns:
            Simplified text
        """
        # Remove some stylistic markers
        text = re.sub(r'\b(notably|significantly|importantly|remarkably)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(very|highly|extremely|particularly)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(We|Our)\b', 'The', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def build_dataset(
        self,
        num_samples: int = 500,
        categories: List[str] = ["cs.CL", "cs.AI", "cs.LG"],
        use_llm_destylization: bool = True
    ):
        """
        Build complete dataset with reference and source texts.

        Args:
            num_samples: Number of paragraph samples to generate
            categories: Arxiv categories to download from
            use_llm_destylization: Use LLM for destylization (vs simple rules)
        """
        print("="*80)
        print("ARXIV STYLE TRANSFER DATASET BUILDER")
        print("="*80)

        # Step 1: Download papers
        print("\n[Step 1/3] Downloading Arxiv papers...")
        # Download more papers to ensure enough paragraphs
        # Since abstracts are ~1 paragraph each, download num_samples * 1.5 papers
        num_papers = int(num_samples * 1.5)
        papers = self.download_arxiv_papers(
            categories=categories,
            num_papers=num_papers
        )

        if not papers:
            print("Error: No papers downloaded. Cannot proceed.")
            return

        # Step 2: Extract paragraphs
        print("\n[Step 2/3] Extracting paragraphs from abstracts...")
        all_paragraphs = []

        for paper in tqdm(papers, desc="Processing papers"):
            paragraphs = self.extract_paragraphs_from_abstract(paper['abstract'])

            for para in paragraphs:
                all_paragraphs.append({
                    'text': para,
                    'paper_id': paper['arxiv_id'],
                    'title': paper['title'],
                    'authors': paper['authors'],
                    'categories': paper['categories']
                })

        print(f"  Extracted {len(all_paragraphs)} paragraphs")

        # Sample paragraphs
        if len(all_paragraphs) > num_samples:
            sampled_paragraphs = random.sample(all_paragraphs, num_samples)
        else:
            sampled_paragraphs = all_paragraphs
            print(f"  Warning: Only {len(all_paragraphs)} paragraphs available (requested {num_samples})")

        # Step 3: Generate source (destylized) texts
        print(f"\n[Step 3/3] Generating source (destylized) texts...")
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
                'title': para_data['title'],
                'authors': para_data['authors'],
                'categories': para_data['categories'],
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
        print(f"Reference texts: {self.reference_dir}")
        print(f"Source texts: {self.source_dir}")
        print(f"Metadata: {self.metadata_file}")
        print("\nNext steps:")
        print("1. Run style transfer models to generate 'transferred' texts")
        print("2. Place transferred texts in: {output_dir}/transferred_{model_name}/")
        print("3. Run evaluation script")

        return dataset

    def load_dataset(self) -> List[Dict]:
        """Load existing dataset metadata."""
        if not self.metadata_file.exists():
            print("No dataset found. Run build_dataset() first.")
            return []

        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_statistics(self):
        """Print dataset statistics."""
        dataset = self.load_dataset()

        if not dataset:
            print("No dataset loaded.")
            return

        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)

        print(f"\nTotal samples: {len(dataset)}")

        # Length statistics
        ref_lengths = [item['reference_length'] for item in dataset]
        src_lengths = [item['source_length'] for item in dataset]

        print(f"\nReference text lengths:")
        print(f"  Mean: {sum(ref_lengths)/len(ref_lengths):.1f} words")
        print(f"  Min: {min(ref_lengths)} words")
        print(f"  Max: {max(ref_lengths)} words")

        print(f"\nSource text lengths:")
        print(f"  Mean: {sum(src_lengths)/len(src_lengths):.1f} words")
        print(f"  Min: {min(src_lengths)} words")
        print(f"  Max: {max(src_lengths)} words")

        # Categories
        all_categories = []
        for item in dataset:
            all_categories.extend(item['categories'])

        from collections import Counter
        category_counts = Counter(all_categories)

        print(f"\nTop categories:")
        for cat, count in category_counts.most_common(10):
            print(f"  {cat}: {count}")


def main():
    """Main function for dataset construction."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Arxiv Style Transfer Dataset")
    parser.add_argument("--output_dir", type=str, default="/root/datasets/arxiv_style_transfer",
                       help="Output directory for dataset")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of paragraph samples to generate")
    parser.add_argument("--categories", type=str, nargs="+",
                       default=["cs.CL", "cs.AI", "cs.LG"],
                       help="Arxiv categories to download from")
    parser.add_argument("--simple_destylize", action="store_true",
                       help="Use simple rule-based destylization (no LLM)")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics of existing dataset")

    args = parser.parse_args()

    # Create builder
    builder = ArxivDatasetBuilder(output_dir=args.output_dir)

    if args.stats:
        builder.get_statistics()
    else:
        # Build dataset
        builder.build_dataset(
            num_samples=args.num_samples,
            categories=args.categories,
            use_llm_destylization=not args.simple_destylize
        )

        # Show statistics
        builder.get_statistics()


if __name__ == "__main__":
    main()
