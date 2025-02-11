def distribute_samples(n_samples, worker_id, n_workers):
    """
    Data distribution function
    Args:
        n_samples: total number of data samples
        worker_id: worker (or NPU) id
        n_workers: number of workers (or NPUs)
    Return:
        (start_idx, end_idx) - the start index and end index (exclusive) for the data samples assinged to worker_id
    """
    base_samples_per_worker = n_samples // n_workers
    remainder = n_samples % n_workers

    distribution = [base_samples_per_worker] * n_workers

    # Distribute the remainder samples
    for i in range(remainder):
        distribution[i] += 1

    # Calculate indices for each node
    indices = []
    start_idx = 0
    for count in distribution:
        end_idx = start_idx + count
        # indices.append(list(range(start_idx, end_idx)))
        indices.append((start_idx, end_idx))
        start_idx = end_idx

    if worker_id is None:
        worker_id = 0

    # get data sample indices for current worker id
    return indices[worker_id]


def data_split_even(samples, worker_id, n_workers):
    """
    Distribute data samples as even as possible
    Args:
        samples: data samples
        worker_id: worker (or NPU) id
        n_workers: number of workers (or NPUs)
    Return:
        data samples assigned to worker_id
    """
    n_samples = len(samples)
    start_idx, end_idx = distribute_samples(n_samples, worker_id, n_workers)

    return samples[start_idx:end_idx]
