import pickle
import logging

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Callable, Any

logger = logging.getLogger(__name__)


def parallel_process(arg_list: List[Dict], function: Callable, n_jobs: int = 1):

    results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:

        futures = [pool.submit(function, **kwargs) for kwargs in arg_list]

        tqdm_kwargs = {
            "total": len(futures),
            "unit": "sims",
            "unit_scale": True,
            "leave": True,
        }

        for f in tqdm(as_completed(futures), **tqdm_kwargs):
            pass

    for i, future in enumerate(futures):
        try:
            results.append(future.result())
        except Exception as e:
            logger.warning(f"Caught exception: {e}")

    return results


def save_to_pickle(data: Any, filepath: str):

    with open(f"{filepath}.pickle", "wb") as f:
        pickle.dump(data, f)
