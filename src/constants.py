from __future__ import annotations

SUPPORTED_METHODS = [
    "full_ft",
    "lora",
    "topheavy_lora",
    "bitfit",
]

SUPPORTED_TASKS = ["sst2", "mrpc", "cola", "qnli", "rte"]

SUPPORTED_MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
]

# Decoder-only models for validation experiments
CAUSAL_MODELS = [
    "Qwen/Qwen2.5-0.5B",
]

# Sample sizes for scaling law experiments
SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000, 5000]

# Reduced sample sizes for Qwen decoder-only validation
CAUSAL_SAMPLE_SIZES = [100, 500, 2000]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}

TASK_PRIMARY_METRIC = {
    "cola": "matthews_correlation",
    "mrpc": "f1",
    "qnli": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
}

DEFAULT_TRAIN_SUBSET = 500
DEFAULT_MAX_LENGTH = 256
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0

# Prompt template for CausalLM classification
CAUSAL_PROMPT_TEMPLATE = (
    "Classify the sentiment of the following sentence as 'positive' or 'negative'.\n"
    "Sentence: {text}\n"
    "Sentiment:"
)
