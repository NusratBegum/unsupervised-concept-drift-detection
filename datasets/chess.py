from os import path

from river.datasets import base
from river import stream


class Chess(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=503,
            n_features=8,
            task=base.MULTI_CLF,
            filename="chess.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        converters = {
            "at1": int,
            "at2": int,
            "at3": int,
            "at4": int,
            "at5": int,
            "at6": int,
            "at7": int,
            "at8": int,
            "label": int,
        }
        return stream.iter_csv(
            self.full_path,
            target="label",
            converters=converters,
        )
