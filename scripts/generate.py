# TODO: write the generate script from a checkpoint
from argparse import ArgumentParser
from threading import local

import torch

from nanogpt.nanogpt_model import NanoGPT, NanoGPTConfig
from nanogpt.training.tokenizer import HuggingFaceTokenizer

if __name__ == "__main__":
    parser = ArgumentParser()

    ### Add ArgumentParser
    parser.add_argument("prompt", type=str, help="Prompt for the model to generate from")
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to the model checkpoint", required=True)
    parser.add_argument(
        "-t", "--max_tokens", type=int, help="Maximum number of tokens to generate. Defaults to 100.", default=100
    )
    parser.add_argument("-T", "--temperature", type=float, help="Temperature for sampling. Defaults to 1.", default=1.0)
    parser.add_argument("-k", "--top_k", type=int, help="Top-k for sampling. Defaults to 50.", default=50)

    args = parser.parse_args()

    # Build config for inference
    prompt = args.prompt
    ckpt_path = args.checkpoint
    n_tokens = args.max_tokens

    print(f"Loading model from checkpoint: {ckpt_path}")
    tokenizer = HuggingFaceTokenizer(tokenizer_name="gpt2")
    local_config = NanoGPTConfig.make_local()
    local_config.vocabulary_size = tokenizer.vocab_size or local_config.vocabulary_size
    model = NanoGPT.load_from_checkpoint(checkpoint_path=ckpt_path, config=local_config)

    tokens: torch.Tensor = tokenizer.encode(prompt).get("input_ids")    # type: ignore

    top_k = min(args.top_k, tokenizer.vocab_size)
    out = model.generate(tokens, n_tokens, temperature=args.temperature, top_k=top_k)

    print(tokenizer.decode(out))
