# Style Transfer Evaluation Report

**Generated:** 2026-01-01 17:28:47

**Models Evaluated:** CAT-LLM, ZeroStylus, DeepTransfer, Zero-Shot, Few-Shot, DT-NoSPECTER, DT-Sentence, DT-NoRerank

---

## Overall Comparison

| Metric | CAT-LLM | ZeroStylus | DeepTransfer | Zero-Shot | Few-Shot | DT-NoSPECTER | DT-Sentence | DT-NoRerank |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| CAT-LLM Overall | 0.5099 | 0.5269 | 0.5990 | 0.5175 | 0.5162 | 0.5917 | 0.5863 | 0.5721 |
| ZeroStylus Average | 5.8284 | 5.7272 | 5.7315 | 5.8348 | 5.8608 | 5.7442 | 5.7574 | 5.7804 |
| Style Accuracy | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9940 | 0.9780 | 0.9980 | 0.9480 |
| BLEU-4 Content | 0.1807 | 0.1323 | 0.1602 | 0.1718 | 0.1832 | 0.1603 | 0.1617 | 0.1603 |
| BLEU-4 Style | 0.1399 | 0.2528 | 0.4554 | 0.1459 | 0.1516 | 0.4364 | 0.3867 | 0.3975 |
| BERTScore F1 Content | 0.7494 | 0.7088 | 0.7241 | 0.7465 | 0.7536 | 0.7238 | 0.7252 | 0.7275 |
| BERTScore F1 Style | 0.7121 | 0.7636 | 0.8312 | 0.7182 | 0.7227 | 0.8240 | 0.8147 | 0.8161 |
| Perplexity | 29.7395 | 27.6392 | 26.5217 | 25.8189 | 28.9070 | 26.1934 | 26.3245 | 31.3223 |
| DT Content Preservation | 0.9761 | 0.9746 | 0.9748 | 0.9766 | 0.9776 | 0.9745 | 0.9758 | 0.9754 |
| DT Style Similarity | 0.8973 | 0.9022 | 0.9118 | 0.7936 | 0.8476 | 0.9120 | 0.8981 | 0.9166 |
| DT Disentanglement | 0.9367 | 0.9384 | 0.9433 | 0.8851 | 0.9126 | 0.9432 | 0.9370 | 0.9460 |


## CAT-LLM Detailed Results

### CAT-LLM Metrics

| Metric | Value |
|--------|-------|
| Bleu-1 Content | 0.5159 |
| Bleu-2 Content | 0.3462 |
| Bleu-3 Content | 0.2459 |
| Bleu-4 Content | 0.1807 |
| Bleu-1 Style | 0.4852 |
| Bleu-2 Style | 0.3036 |
| Bleu-3 Style | 0.2014 |
| Bleu-4 Style | 0.1399 |
| Bert Precision Content | 0.7432 |
| Bert Recall Content | 0.7558 |
| Bert F1 Content | 0.7494 |
| Bert Precision Style | 0.7320 |
| Bert Recall Style | 0.6942 |
| Bert F1 Style | 0.7121 |
| Style Transfer Accuracy | 1.0000 |
| Perplexity Mean | 29.7395 |
| Catllm Overall Score | 0.5099 |

### ZeroStylus Metrics

| Metric | Value |
|--------|-------|
| Style Consistency Mean | 2.4443 |
| Content Preservation Zs Mean | 5.0686 |
| Expression Quality Mean | 9.9722 |
| Zerostylus Average Score | 5.8284 |

### DeepTransfer Metrics

| Metric | Value |
|--------|-------|
| Dt Content Preservation Mean | 0.9761 |
| Dt Style Similarity Mean | 0.8973 |
| Dt Disentanglement Score | 0.9367 |
| Dt Alignment Score Mean | 0.9367 |
| Dt Diversity Pairwise Dist Mean | 0.0557 |


## ZeroStylus Detailed Results

### CAT-LLM Metrics

| Metric | Value |
|--------|-------|
| Bleu-1 Content | 0.4469 |
| Bleu-2 Content | 0.2822 |
| Bleu-3 Content | 0.1888 |
| Bleu-4 Content | 0.1323 |
| Bleu-1 Style | 0.5690 |
| Bleu-2 Style | 0.4113 |
| Bleu-3 Style | 0.3155 |
| Bleu-4 Style | 0.2528 |
| Bert Precision Content | 0.6907 |
| Bert Recall Content | 0.7283 |
| Bert F1 Content | 0.7088 |
| Bert Precision Style | 0.7692 |
| Bert Recall Style | 0.7587 |
| Bert F1 Style | 0.7636 |
| Style Transfer Accuracy | 1.0000 |
| Perplexity Mean | 27.6392 |
| Catllm Overall Score | 0.5269 |

### ZeroStylus Metrics

| Metric | Value |
|--------|-------|
| Style Consistency Mean | 2.5789 |
| Content Preservation Zs Mean | 4.6295 |
| Expression Quality Mean | 9.9733 |
| Zerostylus Average Score | 5.7272 |

### DeepTransfer Metrics

| Metric | Value |
|--------|-------|
| Dt Content Preservation Mean | 0.9746 |
| Dt Style Similarity Mean | 0.9022 |
| Dt Disentanglement Score | 0.9384 |
| Dt Alignment Score Mean | 0.9384 |
| Dt Diversity Pairwise Dist Mean | 0.0410 |


## DeepTransfer Detailed Results

### CAT-LLM Metrics

| Metric | Value |
|--------|-------|
| Bleu-1 Content | 0.4903 |
| Bleu-2 Content | 0.3203 |
| Bleu-3 Content | 0.2214 |
| Bleu-4 Content | 0.1602 |
| Bleu-1 Style | 0.7020 |
| Bleu-2 Style | 0.5883 |
| Bleu-3 Style | 0.5116 |
| Bleu-4 Style | 0.4554 |
| Bert Precision Content | 0.7077 |
| Bert Recall Content | 0.7418 |
| Bert F1 Content | 0.7241 |
| Bert Precision Style | 0.8416 |
| Bert Recall Style | 0.8220 |
| Bert F1 Style | 0.8312 |
| Style Transfer Accuracy | 0.9980 |
| Perplexity Mean | 26.5217 |
| Catllm Overall Score | 0.5990 |

### ZeroStylus Metrics

| Metric | Value |
|--------|-------|
| Style Consistency Mean | 2.5952 |
| Content Preservation Zs Mean | 4.6130 |
| Expression Quality Mean | 9.9863 |
| Zerostylus Average Score | 5.7315 |

### DeepTransfer Metrics

| Metric | Value |
|--------|-------|
| Dt Content Preservation Mean | 0.9748 |
| Dt Style Similarity Mean | 0.9118 |
| Dt Disentanglement Score | 0.9433 |
| Dt Alignment Score Mean | 0.9433 |
| Dt Diversity Pairwise Dist Mean | 0.0545 |


## Zero-Shot Detailed Results

### CAT-LLM Metrics

| Metric | Value |
|--------|-------|
| Bleu-1 Content | 0.5134 |
| Bleu-2 Content | 0.3398 |
| Bleu-3 Content | 0.2374 |
| Bleu-4 Content | 0.1718 |
| Bleu-1 Style | 0.5038 |
| Bleu-2 Style | 0.3166 |
| Bleu-3 Style | 0.2101 |
| Bleu-4 Style | 0.1459 |
| Bert Precision Content | 0.7376 |
| Bert Recall Content | 0.7558 |
| Bert F1 Content | 0.7465 |
| Bert Precision Style | 0.7358 |
| Bert Recall Style | 0.7024 |
| Bert F1 Style | 0.7182 |
| Style Transfer Accuracy | 1.0000 |
| Perplexity Mean | 25.8189 |
| Catllm Overall Score | 0.5175 |

### ZeroStylus Metrics

| Metric | Value |
|--------|-------|
| Style Consistency Mean | 2.5150 |
| Content Preservation Zs Mean | 5.0014 |
| Expression Quality Mean | 9.9879 |
| Zerostylus Average Score | 5.8348 |

### DeepTransfer Metrics

| Metric | Value |
|--------|-------|
| Dt Content Preservation Mean | 0.9766 |
| Dt Style Similarity Mean | 0.7936 |
| Dt Disentanglement Score | 0.8851 |
| Dt Alignment Score Mean | 0.8851 |
| Dt Diversity Pairwise Dist Mean | 0.1229 |


## Few-Shot Detailed Results

### CAT-LLM Metrics

| Metric | Value |
|--------|-------|
| Bleu-1 Content | 0.5196 |
| Bleu-2 Content | 0.3496 |
| Bleu-3 Content | 0.2487 |
| Bleu-4 Content | 0.1832 |
| Bleu-1 Style | 0.4916 |
| Bleu-2 Style | 0.3154 |
| Bleu-3 Style | 0.2138 |
| Bleu-4 Style | 0.1516 |
| Bert Precision Content | 0.7482 |
| Bert Recall Content | 0.7593 |
| Bert F1 Content | 0.7536 |
| Bert Precision Style | 0.7426 |
| Bert Recall Style | 0.7046 |
| Bert F1 Style | 0.7227 |
| Style Transfer Accuracy | 0.9940 |
| Perplexity Mean | 28.9070 |
| Catllm Overall Score | 0.5162 |

### ZeroStylus Metrics

| Metric | Value |
|--------|-------|
| Style Consistency Mean | 2.5623 |
| Content Preservation Zs Mean | 5.0344 |
| Expression Quality Mean | 9.9857 |
| Zerostylus Average Score | 5.8608 |

### DeepTransfer Metrics

| Metric | Value |
|--------|-------|
| Dt Content Preservation Mean | 0.9776 |
| Dt Style Similarity Mean | 0.8476 |
| Dt Disentanglement Score | 0.9126 |
| Dt Alignment Score Mean | 0.9126 |
| Dt Diversity Pairwise Dist Mean | 0.1116 |


## DT-NoSPECTER Detailed Results

### CAT-LLM Metrics

| Metric | Value |
|--------|-------|
| Bleu-1 Content | 0.4885 |
| Bleu-2 Content | 0.3191 |
| Bleu-3 Content | 0.2211 |
| Bleu-4 Content | 0.1603 |
| Bleu-1 Style | 0.6898 |
| Bleu-2 Style | 0.5718 |
| Bleu-3 Style | 0.4933 |
| Bleu-4 Style | 0.4364 |
| Bert Precision Content | 0.7080 |
| Bert Recall Content | 0.7408 |
| Bert F1 Content | 0.7238 |
| Bert Precision Style | 0.8346 |
| Bert Recall Style | 0.8145 |
| Bert F1 Style | 0.8240 |
| Style Transfer Accuracy | 0.9780 |
| Perplexity Mean | 26.1934 |
| Catllm Overall Score | 0.5917 |

### ZeroStylus Metrics

| Metric | Value |
|--------|-------|
| Style Consistency Mean | 2.6071 |
| Content Preservation Zs Mean | 4.6393 |
| Expression Quality Mean | 9.9861 |
| Zerostylus Average Score | 5.7442 |

### DeepTransfer Metrics

| Metric | Value |
|--------|-------|
| Dt Content Preservation Mean | 0.9745 |
| Dt Style Similarity Mean | 0.9120 |
| Dt Disentanglement Score | 0.9432 |
| Dt Alignment Score Mean | 0.9432 |
| Dt Diversity Pairwise Dist Mean | 0.0458 |


## DT-Sentence Detailed Results

### CAT-LLM Metrics

| Metric | Value |
|--------|-------|
| Bleu-1 Content | 0.4952 |
| Bleu-2 Content | 0.3239 |
| Bleu-3 Content | 0.2240 |
| Bleu-4 Content | 0.1617 |
| Bleu-1 Style | 0.6702 |
| Bleu-2 Style | 0.5384 |
| Bleu-3 Style | 0.4506 |
| Bleu-4 Style | 0.3867 |
| Bert Precision Content | 0.7089 |
| Bert Recall Content | 0.7428 |
| Bert F1 Content | 0.7252 |
| Bert Precision Style | 0.8238 |
| Bert Recall Style | 0.8064 |
| Bert F1 Style | 0.8147 |
| Style Transfer Accuracy | 0.9980 |
| Perplexity Mean | 26.3245 |
| Catllm Overall Score | 0.5863 |

### ZeroStylus Metrics

| Metric | Value |
|--------|-------|
| Style Consistency Mean | 2.6112 |
| Content Preservation Zs Mean | 4.6777 |
| Expression Quality Mean | 9.9834 |
| Zerostylus Average Score | 5.7574 |

### DeepTransfer Metrics

| Metric | Value |
|--------|-------|
| Dt Content Preservation Mean | 0.9758 |
| Dt Style Similarity Mean | 0.8981 |
| Dt Disentanglement Score | 0.9370 |
| Dt Alignment Score Mean | 0.9370 |
| Dt Diversity Pairwise Dist Mean | 0.0639 |


## DT-NoRerank Detailed Results

### CAT-LLM Metrics

| Metric | Value |
|--------|-------|
| Bleu-1 Content | 0.4966 |
| Bleu-2 Content | 0.3223 |
| Bleu-3 Content | 0.2222 |
| Bleu-4 Content | 0.1603 |
| Bleu-1 Style | 0.6623 |
| Bleu-2 Style | 0.5402 |
| Bleu-3 Style | 0.4580 |
| Bleu-4 Style | 0.3975 |
| Bert Precision Content | 0.7141 |
| Bert Recall Content | 0.7418 |
| Bert F1 Content | 0.7275 |
| Bert Precision Style | 0.8304 |
| Bert Recall Style | 0.8032 |
| Bert F1 Style | 0.8161 |
| Style Transfer Accuracy | 0.9480 |
| Perplexity Mean | 31.3223 |
| Catllm Overall Score | 0.5721 |

### ZeroStylus Metrics

| Metric | Value |
|--------|-------|
| Style Consistency Mean | 2.6196 |
| Content Preservation Zs Mean | 4.7314 |
| Expression Quality Mean | 9.9901 |
| Zerostylus Average Score | 5.7804 |

### DeepTransfer Metrics

| Metric | Value |
|--------|-------|
| Dt Content Preservation Mean | 0.9754 |
| Dt Style Similarity Mean | 0.9166 |
| Dt Disentanglement Score | 0.9460 |
| Dt Alignment Score Mean | 0.9460 |
| Dt Diversity Pairwise Dist Mean | 0.0503 |


## Key Metrics Comparison

### Content Preservation

| Model | BLEU-4 | BERTScore F1 | DT Content |
|-------|--------|--------------|------------|
| CAT-LLM | 0.1807 | 0.7494 | 0.9761 |
| ZeroStylus | 0.1323 | 0.7088 | 0.9746 |
| DeepTransfer | 0.1602 | 0.7241 | 0.9748 |
| Zero-Shot | 0.1718 | 0.7465 | 0.9766 |
| Few-Shot | 0.1832 | 0.7536 | 0.9776 |
| DT-NoSPECTER | 0.1603 | 0.7238 | 0.9745 |
| DT-Sentence | 0.1617 | 0.7252 | 0.9758 |
| DT-NoRerank | 0.1603 | 0.7275 | 0.9754 |

### Style Transfer

| Model | BLEU-4 Style | Style Accuracy | DT Style Sim |
|-------|--------------|----------------|---------------|
| CAT-LLM | 0.1399 | 1.0000 | 0.8973 |
| ZeroStylus | 0.2528 | 1.0000 | 0.9022 |
| DeepTransfer | 0.4554 | 0.9980 | 0.9118 |
| Zero-Shot | 0.1459 | 1.0000 | 0.7936 |
| Few-Shot | 0.1516 | 0.9940 | 0.8476 |
| DT-NoSPECTER | 0.4364 | 0.9780 | 0.9120 |
| DT-Sentence | 0.3867 | 0.9980 | 0.8981 |
| DT-NoRerank | 0.3975 | 0.9480 | 0.9166 |

