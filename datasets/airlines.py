from os import path

from river.datasets import base
from river import stream


class Airlines(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=539_383,
            n_features=7,
            task=base.MULTI_CLF,
            filename="airlines.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        converters = {
            "Airline": str,
            "Flight": int,
            "AirportFrom": str,
            "AirportTo": str,
            "DayOfWeek": int,
            "Time": int,
            "Length": int,
            "Delay": int,
        }
        return stream.iter_csv(
            self.full_path,
            target="Delay",
            converters=converters,
        )
