import torch

from nanogpt.modules.parallel_multi_head_attention import AttentionConfig
from nanogpt.nanogpt_model import NanoGPT, NanoGPTConfig

if __name__ == "__main__":
    # Some torch housekeeping and consistency
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    # Make tiny config for model
    att_config = AttentionConfig(sequence_length=16, input_dim=8, num_heads=1, hidden_dim=64, dropout=0.1)
    conf = NanoGPTConfig(attention_config=att_config, ff_dims=[8, 8], num_blocks=1)

    # Load the model
    model = NanoGPT(config=conf)

    # Make dummy input
    # dims [B, L]
    t = torch.randint(1, 16, (1, 16), dtype=torch.long)
    print(model(t))
