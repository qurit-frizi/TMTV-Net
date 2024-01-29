from typing import Any
import lz4.frame
import pickle


def read_lz4_pkl(path: str) -> Any:
    with lz4.frame.open(path, mode='rb') as f:
        case_data = pickle.load(f)
    return case_data


def write_lz4_pkl(path: str, data: Any) -> None:
    with lz4.frame.open(path, mode='wb') as f:
        pickle.dump(data, f)