import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from cache.kv_cache import KVCache
from cache.compression_aware import CompressionAwareCache
from compression.svd import svd_compress, svd_decompress
from compression.tucker import tucker_compress, tucker_decompress
from evaluation.perplexity import compute_perplexity
from evaluation.memory import get_memory_usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = (
        r"C:\Users\jylib\.cache\huggingface\hub\models--meta-llama--Meta-Llama-3-8B"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

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

    # Load WikiText dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: tokenizer(
            x, return_tensors="pt", padding=True, truncation=True, max_length=128
        ),
    )

    ppl = compute_perplexity(model, data_loader, device=device)
    mem = get_memory_usage()
    print(f"Perplexity: {ppl:.2f}")
    print(f"Memory usage: {mem}")


if __name__ == "__main__":
    main()
