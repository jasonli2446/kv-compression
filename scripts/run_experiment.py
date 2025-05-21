import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from cache.kv_cache import KVCache
from cache.compression_aware import CompressionAwareCache
from compression.svd import svd_compress, svd_decompress
from compression.tucker import tucker_compress, tucker_decompress
from evaluation.perplexity import compute_perplexity
from evaluation.memory import get_memory_usage
from constants import DEFAULT_MODEL, DEFAULT_DEVICE


def collate_fn(batch, tokenizer, max_length):
    # Combine texts into a single string
    texts = [item["text"] for item in batch]
    # Tokenize the combined texts
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Force use of local model path
    model_name = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"
    device_name = config.get("model", {}).get("device", DEFAULT_DEVICE)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and set padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # Set model's padding token to match tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id

    # Select compression method
    if config["compression"]["type"] == "svd":
        compress_fn = lambda x: svd_compress(x, config["compression"]["rank"])
        decompress_fn = lambda x: svd_decompress(x, x[0][0].shape)
    elif config["compression"]["type"] == "tucker":
        compress_fn = lambda x: tucker_compress(x, config["compression"]["ranks"])
        decompress_fn = lambda x: tucker_decompress(*x)
    else:
        compress_fn = decompress_fn = None

    kv_cache = KVCache(compress_fn, decompress_fn)
    cache = CompressionAwareCache(kv_cache)

    # Load dataset with settings from config
    dataset_config = config.get("dataset", {})
    dataset = load_dataset(
        dataset_config.get("name", "wikitext"),
        dataset_config.get("config", "wikitext-2-raw-v1"),
        split=dataset_config.get("split", "train"),
    )

    # Create data loader with proper collate function
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_config.get("batch_size", 4),
        shuffle=True,
        collate_fn=lambda x: collate_fn(
            x, tokenizer, dataset_config.get("max_length", 128)
        ),
    )

    ppl = compute_perplexity(model, data_loader, device=device)
    mem = get_memory_usage()
    print(f"Perplexity: {ppl:.2f}")
    print(f"Memory usage: {mem}")


if __name__ == "__main__":
    main()
