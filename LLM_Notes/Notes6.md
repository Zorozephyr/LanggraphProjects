# LLM Comparison & Benchmarking

## Key LLM Comparison Factors
- Model Type: Open source vs closed
- Knowledge Cut-off & Release Date
- Parameter Count
- Training Tokens & Build Cost
- Context Length
- Inference & Training Cost
- Time to Market & Rate Limits
- Speed & Latency
- License

## Scaling Principles
**Chinchilla Scaling Law**: Parameter count should be proportional to training tokens. Doubling model size requires proportional data increase.

## Benchmark Categories

### Common Benchmarks (7)
| Benchmark | Focus |
|-----------|-------|
| ARC | Reasoning |
| DROP | Language Comprehension |
| HellaSwag | Common Sense |
| MMLU | Understanding |
| TruthfulQA | Factual Accuracy |
| Winogrande | Contextual Understanding |
| GSM8K | Mathematical Reasoning |

### Specialized Benchmarks (3)
- **ELO**: Chat quality (head-to-head comparisons)
- **HumanEval**: Python coding
- **MultiPL-E**: Broader coding capabilities

### Advanced Benchmarks (6)
- **GPQA**: Graduate-level questions (448 expert Qs; non-PhDs score 34% with web access)
- **BBHard**: Future AI capabilities
- **Math Lv 5**: Advanced mathematics
- **IFEval**: Instruction-following under constraints
- **MuSR**: Multistep soft reasoning
- **MMLU-PRO**: Extended MMLU

## Benchmark Limitations
1. Inconsistent application across models
2. Narrow scope doesn't capture full capabilities
3. Hard to measure nuanced reasoning
4. Training data leakage—difficult to prevent test question inclusion
5. Overfitting to benchmarks
6. Frontier models may be aware of evaluation (unproven)

## Resources
- LLM-Perf Leaderboard
- LMSYS Chatbot Arena