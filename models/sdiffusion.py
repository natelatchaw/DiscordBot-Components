from typing import List, Tuple

from torch import Size


class StableDiffusionSettings():

    def __init__(self, *, dimensions: Tuple[int, int], downsampling: int, batch_size: int, scale: float, channels: int, eta: float, seed: int):
        self._height: int = dimensions[0]
        self._width: int = dimensions[1]
        self._downsampling: int = downsampling
        self._batch_size: int = batch_size
        self._scale: float = scale
        self._channels: int = channels
        self._eta: float = eta
        self._seed: int = seed

    @property
    def height(self) -> int: return self._height

    @property
    def width(self) -> int: return self._width

    @property
    def downsampling(self) -> int: return self._downsampling

    @property
    def batch_size(self) -> int: return self._batch_size

    @property
    def scale(self) -> float: return self._scale

    @property
    def channels(self) -> int: return self._channels

    @property
    def eta(self) -> float: return self._eta

    @property
    def seed(self) -> int: return self._seed

    @property
    def shape(self) -> List[int]: return [
        self._channels,
        self._height // self._downsampling,
        self._width // self._downsampling
    ]

    @property
    def size(self) -> Size: return Size([
        self._batch_size,
        self._channels,
        self._height // self._downsampling,
        self._width // self._downsampling
    ])

class StableDiffusionPresets():

    @staticmethod
    def low(seed: int) -> StableDiffusionSettings:
        return StableDiffusionSettings(
            dimensions=(256, 256),
            downsampling=8,
            batch_size=1,
            scale=4.0,
            channels=4,
            eta=0.75,
            seed=seed
        )

    @staticmethod
    def medium(seed: int) -> StableDiffusionSettings:
        return StableDiffusionSettings(
            dimensions=(380, 380),
            downsampling=8,
            batch_size=1,
            scale=7.5,
            channels=4,
            eta=0.8,
            seed=seed
        )

    @staticmethod
    def high(seed: int) -> StableDiffusionSettings:
        return StableDiffusionSettings(
            dimensions=(448, 448),
            downsampling=8,
            batch_size=1,
            scale=8.0,
            channels=4,
            eta=0.85,
            seed=seed
        )
