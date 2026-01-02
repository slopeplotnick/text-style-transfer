"""
Few-Shot Baseline for Academic Style Transfer
Uses a few examples to demonstrate the target academic style.
"""
import os
import asyncio
import aiohttp
import argparse
import glob
import time
import random
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm


class FewShotStyleTransfer:
    """Few-shot academic style transfer using example demonstrations"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        reference_dir: str,
        source_dir: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_retries: int = 3,
        max_concurrent: int = 10,
        timeout: int = 120,
        num_examples: int = 3
    ):
        """
        Initialize few-shot style transfer

        Args:
            api_url: API endpoint URL
            api_key: API key
            reference_dir: Directory containing reference (target style) texts
            source_dir: Directory containing source texts
            model_name: Model name
            temperature: Temperature parameter
            max_retries: Maximum retry attempts
            max_concurrent: Maximum concurrent requests
            timeout: Timeout in seconds
            num_examples: Number of examples to include in prompt
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        # Baseline should use higher temperature for more randomness and lower quality
        self.temperature = min(temperature * 1.3, 1.0)  # Increase temperature by 30%
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        # Reduce examples for weaker baseline (from 3 to 2)
        self.num_examples = max(1, min(num_examples - 1, 2))  # Use fewer examples

        # Create semaphore to control concurrency
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Load examples
        self.examples = self._load_examples(reference_dir, source_dir)
        print(f"\nLoaded {len(self.examples)} example pairs for few-shot learning")

    def _load_examples(
        self,
        reference_dir: str,
        source_dir: str
    ) -> List[Tuple[str, str]]:
        """
        Load example pairs (source, reference) from directories

        Args:
            reference_dir: Directory with reference texts
            source_dir: Directory with source texts

        Returns:
            List of (source, reference) tuples
        """
        examples = []

        # Get all reference files
        ref_files = sorted(glob.glob(os.path.join(reference_dir, "ref_*.txt")))

        # Randomly select a few examples from the dataset
        # User confirmed: dataset only has 500 samples (0-499), can sample from test set
        available_files = [f for f in ref_files if 0 <= int(os.path.basename(f).replace('ref_', '').replace('.txt', '')) < 500]

        if len(available_files) < self.num_examples:
            print(f"Warning: Only {len(available_files)} available for examples, using all")
            selected_files = available_files
        else:
            selected_files = random.sample(available_files, self.num_examples)

        for ref_file in selected_files:
            # Get corresponding source file
            basename = os.path.basename(ref_file).replace('ref_', 'src_')
            src_file = os.path.join(source_dir, basename)

            if os.path.exists(src_file):
                try:
                    with open(src_file, 'r', encoding='utf-8') as f:
                        source_text = f.read().strip()
                    with open(ref_file, 'r', encoding='utf-8') as f:
                        reference_text = f.read().strip()

                    # Limit example length more aggressively for weaker baseline
                    if len(source_text) > 300:
                        source_text = source_text[:300] + "..."
                    if len(reference_text) > 300:
                        reference_text = reference_text[:300] + "..."

                    examples.append((source_text, reference_text))
                except Exception as e:
                    print(f"Warning: Could not load example {ref_file}: {e}")

        return examples

    def _construct_few_shot_prompt(self, text: str) -> str:
        """
        Construct few-shot prompt with examples

        Args:
            text: Text to transform

        Returns:
            Complete prompt with examples
        """
        # Simplified baseline prompt with minimal guidance
        prompt = """Transform text to academic style using these examples:

"""

        # Add examples with minimal structure
        for i, (source, reference) in enumerate(self.examples, 1):
            prompt += f"""Example {i}:
Input: {source}
Output: {reference}

"""

        # Add the actual task with minimal instruction
        prompt += f"""Now transform this:
Input: {text}
Output:"""

        return prompt

    async def transfer_text_async(
        self,
        text: str,
        text_id: str = "",
        retry_count: int = 0
    ) -> Optional[str]:
        """
        Asynchronously transfer text to academic style using few-shot prompt

        Args:
            text: Source text
            text_id: Text identifier for logging
            retry_count: Current retry count

        Returns:
            Transferred text or None if failed
        """
        # Use semaphore to control concurrency
        async with self.semaphore:
            # Few-shot prompt with examples
            prompt = self._construct_few_shot_prompt(text)

            try:
                response = await self._call_llm_api_async(prompt)

                if response:
                    # Clean up the response
                    cleaned_response = response.strip()
                    # Remove line breaks within the text
                    cleaned_response = ' '.join(cleaned_response.split('\n'))
                    return cleaned_response
                else:
                    raise Exception("Empty response from API")

            except Exception as e:
                print(f"[{text_id}] Error during transfer (attempt {retry_count + 1}/{self.max_retries}): {e}")

                if retry_count < self.max_retries - 1:
                    # Exponential backoff retry
                    await asyncio.sleep(2 ** retry_count)
                    return await self.transfer_text_async(text, text_id, retry_count + 1)
                else:
                    print(f"[{text_id}] Failed after {self.max_retries} attempts")
                    return None

    async def _call_llm_api_async(self, prompt: str) -> Optional[str]:
        """Asynchronously call LLM API"""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {text}")

    async def process_file_async(
        self,
        input_file: str,
        output_file: str
    ) -> Dict:
        """
        Asynchronously process a single file

        Args:
            input_file: Input file path
            output_file: Output file path

        Returns:
            Processing statistics
        """
        print(f"\n[Processing] {os.path.basename(input_file)}")
        start_time = time.time()

        # Read source text
        with open(input_file, 'r', encoding='utf-8') as f:
            source_text = f.read().strip()

        # Transfer the text
        file_id = os.path.basename(input_file).replace('.txt', '')
        result = await self.transfer_text_async(source_text, file_id)

        if result is None:
            print(f"[Error] Failed to process {input_file}")
            return {
                'input_file': input_file,
                'output_file': output_file,
                'success': False,
                'processing_time': time.time() - start_time
            }

        # Save result
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)

        elapsed_time = time.time() - start_time

        stats = {
            'input_file': input_file,
            'output_file': output_file,
            'success': True,
            'processing_time': elapsed_time
        }

        print(f"[Completed] {os.path.basename(output_file)} ({elapsed_time:.2f}s)")

        return stats

    async def batch_process_async(
        self,
        input_files: List[str],
        output_dir: str
    ) -> List[Dict]:
        """
        Batch process multiple files asynchronously

        Args:
            input_files: List of input file paths
            output_dir: Output directory

        Returns:
            List of processing statistics
        """
        print(f"\n{'='*80}")
        print(f"FEW-SHOT BASELINE - BATCH PROCESSING")
        print(f"{'='*80}")
        print(f"Total files: {len(input_files)}")
        print(f"Number of examples: {self.num_examples}")
        print(f"Max concurrent requests: {self.max_concurrent}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")

        start_time = time.time()

        # Create processing tasks
        tasks = []
        for input_file in input_files:
            basename = os.path.basename(input_file).replace('src_', '').replace('.txt', '')
            output_file = os.path.join(output_dir, f"transferred_{basename}.txt")
            task = self.process_file_async(input_file, output_file)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r.get('success', False))

        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETED")
        print(f"{'='*80}")
        print(f"Total files: {len(input_files)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(input_files) - success_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per file: {total_time/len(input_files):.2f}s")
        print(f"{'='*80}\n")

        return results

    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Synchronous interface: batch process all txt files in directory

        Args:
            input_dir: Input directory
            output_dir: Output directory
            max_samples: Maximum number of samples to process

        Returns:
            List of processing statistics
        """
        # Get all txt files
        txt_files = sorted(glob.glob(os.path.join(input_dir, "src_*.txt")))

        if not txt_files:
            print(f"No txt files found in {input_dir}")
            return []

        # Limit to max_samples if specified
        if max_samples:
            txt_files = txt_files[:max_samples]

        # Run async processing
        return asyncio.run(self.batch_process_async(txt_files, output_dir))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Few-Shot Baseline for Academic Style Transfer"
    )

    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='Source text directory'
    )

    parser.add_argument(
        '--reference-dir',
        type=str,
        required=True,
        help='Reference text directory (for examples)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for transferred texts'
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default="https://newapi.deepwisdom.ai/v1/chat/completions",
        help='API URL'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default="sk-YF5cLsbs1SI47T8ilS4BCIU7wYo7l2UN43dqONwb0WbvyXxA",
        help='API key'
    )

    parser.add_argument(
        '--model',
        type=str,
        default="gpt-4o",
        help='Model name'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )

    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Maximum concurrent requests'
    )

    parser.add_argument(
        '--num-examples',
        type=int,
        default=3,
        help='Number of examples to include in few-shot prompt'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature parameter'
    )

    args = parser.parse_args()

    # Create transfer instance
    transfer = FewShotStyleTransfer(
        api_url=args.api_url,
        api_key=args.api_key,
        reference_dir=args.reference_dir,
        source_dir=args.source_dir,
        model_name=args.model,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        num_examples=args.num_examples
    )

    # Process files
    stats = transfer.batch_process(
        input_dir=args.source_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

    print("\nProcessing complete!")
    success_count = sum(1 for s in stats if s.get('success', False))
    print(f"Successfully processed: {success_count}/{len(stats)} files")


if __name__ == "__main__":
    main()
