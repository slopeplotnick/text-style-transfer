"""
Test script for split_into_sentences function
"""

def test_split_sentences():
    """Test the split_into_sentences function."""

    print("=" * 70)
    print("Testing split_into_sentences Function")
    print("=" * 70)

    # Step 1: Check NLTK availability
    print("\n[1/4] Checking NLTK installation...")
    try:
        import nltk
        print(f"✓ NLTK version: {nltk.__version__}")
    except ImportError:
        print("✗ NLTK not installed!")
        print("Install with: pip install nltk")
        return False

    # Step 2: Check punkt tokenizer
    print("\n[2/4] Checking punkt tokenizer...")
    try:
        nltk.data.find('tokenizers/punkt')
        print("✓ punkt tokenizer already downloaded")
    except LookupError:
        print("⚠ punkt tokenizer not found, downloading...")
        try:
            nltk.download('punkt')
            print("✓ punkt tokenizer downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download punkt tokenizer: {e}")
            print("\nManual download:")
            print("  python -m nltk.downloader punkt")
            return False

    # Step 3: Import the function
    print("\n[3/4] Importing split_into_sentences...")
    try:
        from utils import split_into_sentences
        print("✓ Function imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

    # Step 4: Test the function
    print("\n[4/4] Testing function with sample text...")

    test_cases = [
        {
            "name": "Simple text",
            "text": "This is the first sentence. This is the second sentence. And this is the third.",
            "expected_count": 3
        },
        {
            "name": "Complex text with abbreviations",
            "text": "Dr. Smith works at MIT. He has a Ph.D. in Computer Science. He loves AI research.",
            "expected_count": 3
        },
        {
            "name": "Academic text",
            "text": "Machine learning has revolutionized data analysis. Deep learning models achieve state-of-the-art results. However, interpretability remains a challenge.",
            "expected_count": 3
        },
        {
            "name": "Single sentence",
            "text": "This is just one sentence.",
            "expected_count": 1
        },
        {
            "name": "Empty text",
            "text": "",
            "expected_count": 0
        }
    ]

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Input: \"{test['text'][:50]}{'...' if len(test['text']) > 50 else ''}\"")

        try:
            sentences = split_into_sentences(test['text'])
            count = len(sentences)

            print(f"Output: {count} sentence(s)")

            if sentences:
                for j, sent in enumerate(sentences, 1):
                    print(f"  [{j}] {sent}")

            if count == test['expected_count']:
                print("✓ PASSED")
            else:
                print(f"✗ FAILED: Expected {test['expected_count']}, got {count}")
                all_passed = False

        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests PASSED!")
        print("split_into_sentences is working correctly.")
    else:
        print("✗ Some tests FAILED!")
        print("Please check the errors above.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = test_split_sentences()
    exit(0 if success else 1)
