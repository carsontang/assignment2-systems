import argparse
import logging
import sys
import torch
from dataclasses import dataclass
import time
import torch.nn as nn
import torch.nn.functional as F

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
    n_warmups: int = 3
    n_runs: int = 10

def benchmark(config, model, n_warmup, n_runs):
    device = next(model.parameters()).device
    logging.info(f'Running benchmarks with model on device={device}')

    def warmup():
        times = []
        for _ in range(n_warmup):
            x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
            start_time = time.perf_counter()
            with torch.no_grad():
                model(x) # fwd pass
            delta = time.perf_counter() - start_time
            times.append(delta)
        return times

    def forward_pass():
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
        start_time = time.perf_counter()
        with torch.no_grad():
            model(x)
        delta = time.perf_counter() - start_time
        return delta
    
    def backward_pass():
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
        loss_fn = nn.CrossEntropyLoss()
        logits = model(x)
        last_logit = logits[:, -1, :]
        targets = torch.randint(0, config.vocab_size, (config.batch_size,), dtype=torch.long, device=device)
        # logging.info(f'{logits.shape=}')
        # logging.info(f'{targets.shape=}')
        loss = loss_fn(input=last_logit, target=targets)
        start = time.perf_counter()
        loss.backward()
        delta = time.perf_counter() - start
        return delta

    
    def warmup_cuda():
        times = []
        for _ in range(n_warmup):
            x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
            start_time = time.perf_counter()
            with torch.no_grad():
                model(x)
            torch.cuda.synchronize()
            delta = time.perf_counter() - start_time
            times.append(delta)
        return times
    
    def forward_pass_cuda():
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
        start_time = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        delta = time.perf_counter() - start_time
        return delta
    
    def backward_pass_cuda():
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
        loss_fn = nn.CrossEntropyLoss()
        logits = model(x)
        last_logit = logits[:, -1, :]
        targets = torch.randint(0, config.vocab_size, (config.batch_size,), dtype=torch.long, device=device)
        loss = loss_fn(input=last_logit, target=targets)
        start_time = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        delta = time.perf_counter() - start_time
        return delta
    
    
    if device.type == 'cpu':
        logging.info('Timing forward pass on CPU')
        warmups = []
        times = []
        warmups.extend(warmup())
        for _ in range(n_runs):
            times.append(forward_pass())
        
        logging.info('Timing backward pass on CPU')
        bwd_warmup_times = []
        bwd_times = []
        for _ in range(n_warmup):
            bwd_warmup_times.append(backward_pass())

        for _ in range(n_runs):
            bwd_times.append(backward_pass())
    else:
        logging.info('Timing forward pass on GPU')
        warmups = []
        times = []
        warmups.extend(warmup_cuda())
        for _ in range(n_runs):
            times.append(forward_pass_cuda())
        
        logging.info('Timing backward pass on GPU')
        bwd_warmup_times = []
        bwd_times = []
        for _ in range(n_warmup):
            bwd_warmup_times.append(backward_pass_cuda())

        for _ in range(n_runs):
            bwd_times.append(backward_pass_cuda())
    
    avg = sum(times) / len(times)
    maxtime = max(times)
    mintime = min(times)

    logging.info(f'{avg=:.5f}')
    logging.info(f'{maxtime=:.5f}')
    logging.info(f'{mintime=:.5f}')
    logging.info([f'{w:.5f}' for w in warmups])
    logging.info([f'{t:.5f}' for t in times])

    avg = sum(bwd_times) / len(bwd_times)
    maxtime = max(bwd_times)
    mintime = min(bwd_times)

    logging.info(f'{avg=:.5f}')
    logging.info(f'{maxtime=:.5f}')
    logging.info(f'{mintime=:.5f}')
    logging.info([f'{w:.5f}' for w in bwd_warmup_times])
    logging.info([f'{t:.5f}' for t in bwd_times])


def build_config_message(config: Config):
    """Build a formatted configuration message for logging."""
    config_parts = [
        f"batch_size={config.batch_size}",
        f"vocab_size={config.vocab_size:,}",
        f"context_length={config.context_length}",
        f"d_model={config.d_model}",
        f"num_layers={config.num_layers}",
        f"num_heads={config.num_heads}",
        f"d_ff={config.d_ff}",
        f"n_warmups={config.n_warmups}",
        f"n_runs={config.n_runs}",
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
    parser.add_argument('--num-runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--num-warmup', type=int, default=2, help='Number of warmup runs')

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
        use_random_seed=args.use_random_seed,
        n_warmups=args.num_warmup,
        n_runs=args.num_runs,
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

    # move model to CUDA if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    logging.info(f'Using device: {device}')

    # Benchmark the model
    # timing_stats = benchmark_model(model, x, num_runs=args.num_runs, num_warmup=args.num_warmup)
    benchmark(config, model, n_warmup=args.num_warmup, n_runs=args.num_runs)
    
    # # Log results
    # logging.info(f"Average time: {timing_stats['avg']:.4f} seconds")
    # logging.info(f"Min time: {timing_stats['min']:.4f} seconds")
    # logging.info(f"Max time: {timing_stats['max']:.4f} seconds")
    # logging.info(f"All times: {[f'{t:.4f}' for t in timing_stats['all_times']]}")


if __name__ == '__main__':
    main()