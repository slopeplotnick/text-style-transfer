"""
Data Loader for Unified Style Transfer Testing
Handles loading of reference, source, and transferred texts
"""
import os
import json
import glob
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Sample:
    """Single test sample containing all text variants"""
    id: int
    paper_id: str
    reference: str
    source: str
    transferred: Optional[str] = None
    reference_file: str = ""
    source_file: str = ""
    transferred_file: str = ""


class DataLoader:
    """Loads and manages test data for style transfer evaluation"""

    def __init__(self, reference_dir: str, source_dir: str, metadata_file: str):
        """
        Initialize data loader.

        Args:
            reference_dir: Directory containing reference texts
            source_dir: Directory containing source texts
            metadata_file: Path to metadata JSON file
        """
        self.reference_dir = reference_dir
        self.source_dir = source_dir
        self.metadata_file = metadata_file
        self.metadata = None

        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata with {len(self.metadata)} samples")
        else:
            print(f"Warning: Metadata file not found at {self.metadata_file}")
            self.metadata = []

    def _read_text_file(self, file_path: str) -> str:
        """Read text from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def load_base_samples(self, max_samples: Optional[int] = None) -> List[Sample]:
        """
        Load reference and source samples.

        Args:
            max_samples: Maximum number of samples to load (None = all)

        Returns:
            List of Sample objects with reference and source filled
        """
        samples = []

        metadata_to_use = self.metadata[:max_samples] if max_samples else self.metadata

        for item in metadata_to_use:
            sample_id = item['id']
            paper_id = item.get('paper_id', f'sample_{sample_id}')
            ref_file = item['reference_file']
            src_file = item['source_file']

            ref_path = os.path.join(self.reference_dir, ref_file)
            src_path = os.path.join(self.source_dir, src_file)

            reference_text = self._read_text_file(ref_path)
            source_text = self._read_text_file(src_path)

            if reference_text and source_text:
                sample = Sample(
                    id=sample_id,
                    paper_id=paper_id,
                    reference=reference_text,
                    source=source_text,
                    reference_file=ref_file,
                    source_file=src_file
                )
                samples.append(sample)

        print(f"Loaded {len(samples)} base samples (reference + source)")
        return samples

    def load_transferred_texts(
        self,
        samples: List[Sample],
        transferred_dir: str,
        naming_pattern: str = "auto"
    ) -> List[Sample]:
        """
        Load transferred texts and match them to samples.

        Args:
            samples: List of Sample objects to populate
            transferred_dir: Directory containing transferred texts
            naming_pattern: How to match files
                - "auto": Try to auto-detect pattern
                - "ref_XXXX": Match reference file naming (ref_0000.txt -> ref_0000.txt)
                - "src_XXXX": Match source file naming (src_0000.txt -> src_0000.txt)
                - "paper_id": Match by paper_id (2512.03025v1.txt)
                - "sequential": Match by ID order (0.txt, 1.txt, ...)

        Returns:
            Updated samples with transferred texts
        """
        if not os.path.exists(transferred_dir):
            print(f"Warning: Transferred directory not found: {transferred_dir}")
            return samples

        print(f"Loading transferred texts from {transferred_dir}")

        # Get all text files in directory
        txt_files = glob.glob(os.path.join(transferred_dir, "*.txt"))
        if not txt_files:
            print(f"Warning: No .txt files found in {transferred_dir}")
            return samples

        print(f"Found {len(txt_files)} transferred files")

        # Auto-detect pattern if needed
        if naming_pattern == "auto":
            naming_pattern = self._detect_naming_pattern(txt_files, samples)
            print(f"Auto-detected naming pattern: {naming_pattern}")

        # Match files to samples
        matched_count = 0
        for sample in samples:
            transferred_path = self._find_transferred_file(
                sample, transferred_dir, txt_files, naming_pattern
            )

            if transferred_path:
                transferred_text = self._read_text_file(transferred_path)
                if transferred_text:
                    sample.transferred = transferred_text
                    sample.transferred_file = os.path.basename(transferred_path)
                    matched_count += 1

        print(f"Matched {matched_count}/{len(samples)} transferred texts to samples")
        return samples

    def _detect_naming_pattern(self, txt_files: List[str], samples: List[Sample]) -> str:
        """Auto-detect the naming pattern of transferred files"""
        basenames = [os.path.basename(f) for f in txt_files]

        # Check for transferred_XXXX pattern (unified output format)
        if any(name.startswith('transferred_') for name in basenames):
            return "transferred_XXXX"

        # Check for ref_XXXX pattern
        if any(name.startswith('ref_') for name in basenames):
            return "ref_XXXX"

        # Check for src_XXXX pattern
        if any(name.startswith('src_') for name in basenames):
            return "src_XXXX"

        # Check for paper_id pattern (contains 'v')
        if samples and any(samples[0].paper_id in name for name in basenames):
            return "paper_id"

        # Check for sequential pattern (0.txt, 1.txt, etc.)
        if any(name.split('.')[0].isdigit() for name in basenames):
            return "sequential"

        return "ref_XXXX"  # Default fallback

    def _find_transferred_file(
        self,
        sample: Sample,
        transferred_dir: str,
        txt_files: List[str],
        pattern: str
    ) -> Optional[str]:
        """Find the transferred file for a sample based on naming pattern"""

        if pattern == "transferred_XXXX":
            # Format: transferred_0000.txt matching sample.id
            target_name = f"transferred_{sample.id:04d}.txt"
        elif pattern == "ref_XXXX":
            target_name = sample.reference_file
        elif pattern == "src_XXXX":
            target_name = sample.source_file
        elif pattern == "paper_id":
            target_name = f"{sample.paper_id}.txt"
        elif pattern == "sequential":
            target_name = f"{sample.id}.txt"
        else:
            target_name = sample.reference_file

        target_path = os.path.join(transferred_dir, target_name)

        if os.path.exists(target_path):
            return target_path

        # Fallback: search by filename in the list
        for file_path in txt_files:
            if os.path.basename(file_path) == target_name:
                return file_path

        return None

    def filter_valid_samples(self, samples: List[Sample]) -> List[Sample]:
        """Filter samples that have all required texts (reference, source, transferred)"""
        valid_samples = [s for s in samples if s.reference and s.source and s.transferred]
        print(f"Filtered to {len(valid_samples)} valid samples (all texts present)")
        return valid_samples

    def extract_texts(self, samples: List[Sample]) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract separate lists of texts for evaluation.

        Args:
            samples: List of Sample objects

        Returns:
            Tuple of (references, sources, transferred)
        """
        references = [s.reference for s in samples]
        sources = [s.source for s in samples]
        transferred = [s.transferred for s in samples]

        return references, sources, transferred

    def save_samples_summary(self, samples: List[Sample], output_file: str):
        """Save a summary of loaded samples to JSON"""
        summary = []
        for sample in samples:
            summary.append({
                'id': sample.id,
                'paper_id': sample.paper_id,
                'reference_file': sample.reference_file,
                'source_file': sample.source_file,
                'transferred_file': sample.transferred_file,
                'has_reference': bool(sample.reference),
                'has_source': bool(sample.source),
                'has_transferred': bool(sample.transferred),
                'reference_length': len(sample.reference) if sample.reference else 0,
                'source_length': len(sample.source) if sample.source else 0,
                'transferred_length': len(sample.transferred) if sample.transferred else 0
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Saved samples summary to {output_file}")


def load_model_data(
    reference_dir: str,
    source_dir: str,
    metadata_file: str,
    transferred_dir: str,
    max_samples: Optional[int] = None,
    naming_pattern: str = "auto"
) -> Tuple[List[str], List[str], List[str], List[Sample]]:
    """
    Convenience function to load all data for a model.

    Returns:
        Tuple of (references, sources, transferred, samples)
    """
    loader = DataLoader(reference_dir, source_dir, metadata_file)

    # Load base samples
    samples = loader.load_base_samples(max_samples)

    # Load transferred texts
    samples = loader.load_transferred_texts(samples, transferred_dir, naming_pattern)

    # Filter valid samples
    samples = loader.filter_valid_samples(samples)

    # Extract texts
    references, sources, transferred = loader.extract_texts(samples)

    return references, sources, transferred, samples
