# metrics/metrics.py
def speedup(t_serial, t_parallel):
    return t_serial / t_parallel


def efficiency(speedup, p):
    return speedup / p


def throughput(N, K, D, n_iters, total_time):
    return (N * K * D * n_iters) / total_time
