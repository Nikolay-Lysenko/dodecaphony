"""
Do auxiliary tasks.

Author: Nikolay Lysenko
"""


import multiprocessing as mp
from typing import Any, Callable, Optional


def starmap_in_parallel(
        fn: Callable, args: list[Any], pool_kwargs: Optional[dict[str, Any]] = None
) -> list[Any]:
    """
    Apply function to each collection of arguments from the given list in parallel.

    This function contains boilerplate code that is needed for correct work
    of `pytest-cov`. Usage of `mp.Pool` as context manager is not alternative
    to this function, because:
    1) not all covered lines of code may be marked as covered
       (see more: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html);
    2) some files with names like '.coverage.hostname.*' may be not deleted
       (see more: https://github.com/pytest-dev/pytest-cov/issues/250).

    :param fn:
        function
    :param args:
        list of collections of arguments
    :param pool_kwargs:
        parameters of pool such as number of processes and maximum number of
        tasks for a worker before it is replaced with a new one
    :return:
        results of applying the function to the arguments
    """
    pool_kwargs = pool_kwargs or {}
    key_renaming = {'n_processes': 'processes', 'max_tasks_per_child': 'maxtasksperchild'}
    pool_kwargs = {key_renaming.get(k, k): v for k, v in pool_kwargs.items()}
    pool = mp.Pool(**pool_kwargs)
    try:
        results = pool.starmap(fn, args)
    finally:
        pool.close()
        pool.join()
    return results


def compute_rolling_aggregate(
        values: list[float], aggregation_fn: Callable[[list[float]], float], window_size: int
) -> list[float]:
    """
    Compute rolling aggregate.

    :param values:
        list of values to be aggregated
    :param aggregation_fn:
        aggregation function
    :param window_size:
        size of rolling window
    :return:
        list of rolling aggregates
    """
    window = []
    results = []
    for value in values:
        if len(window) == window_size:
            window.pop(0)
        window.append(value)
        results.append(aggregation_fn(window))
    return results
