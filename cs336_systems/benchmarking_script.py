import argparse
import logging
import sys
import torch
from dataclasses import dataclass
import time
from contextlib import contextmanager

from cs336_basics.model import BasicsTransformerLM

log_format = "[%(asctime)s] [%(levelname)s] %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@dataclass
class Config:
    batch_size: int = 32
    vocab_size: int = 10_000
    context_length: int = 32
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0
    use_random_seed: bool = False

@contextmanager
def timer():
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    diff = end_time - start_time
    logging.info(f'Took {diff:.4f} seconds')

def build_config_message(config: Config):
    """Build a formatted configuration message for logging."""
    config_parts = [
        f"batch_size={config.batch_size}",
        f"vocab_size={config.vocab_size:,}",
        f"context_length={config.context_length}",
        f"d_model={config.d_model}",
        f"num_layers={config.num_layers}",
        f"num_heads={config.num_heads}",
        f"d_ff={config.d_ff:,}"
    ]
    return "\nModel Configuration: " + "\n\t".join(config_parts)

def generate_random_data(config: Config):
    random_data = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
    return random_data

def main():
    parser = argparse.ArgumentParser(description="Benchmark model")
    parser.add_argument('-b', '--batchsize', type=int, default=32)
    parser.add_argument('-v', '--vocab', type=int, default=10_000)
    parser.add_argument('-c', '--context', type=int, default=32)
    parser.add_argument('-e', '--dmodel', type=int, default=768)
    parser.add_argument('-l', '--nlayers', type=int, default=12)
    parser.add_argument('-nh', '--nheads', type=int, default=12)
    parser.add_argument('-f', '--dff', type=int, default=3072)
    parser.add_argument('--use-random-seed', action='store_true')

    args = parser.parse_args()
    
    # Create config object from arguments
    config = Config(
        batch_size=args.batchsize,
        vocab_size=args.vocab,
        context_length=args.context,
        d_model=args.dmodel,
        num_layers=args.nlayers,
        num_heads=args.nheads,
        d_ff=args.dff,
        use_random_seed=args.use_random_seed
    )

    if config.use_random_seed:
        logging.info('Setting random seed to 1337')
        torch.manual_seed(1337)

    summary_parts = build_config_message(config)
    logging.info(summary_parts)
    
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size, 
        context_length=config.context_length, 
        d_model=config.d_model, 
        num_layers=config.num_layers, 
        num_heads=config.num_heads, 
        d_ff=config.d_ff,
        rope_theta=config.rope_theta
    )

    x = generate_random_data(config)
    logging.info(f'passing x ({x.shape}) through model')
    with timer():
        out = model(x)


if __name__ == '__main__':
    main()