from os import PathLike

from t2i.stable_diffusion import StableDiffusionGenerator
from t2i.text_to_image_item import TextToImageItem


class GenerateImageUsecase:
    def __init__(self, generator: StableDiffusionGenerator):
        self.generator = generator

    def generate(self, item: TextToImageItem) -> PathLike:
        path = self.generator.generate(item)
        return path
