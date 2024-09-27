from os import PathLike

from t2i import env
from t2i.generate_image_usecase import GenerateImageUsecase
from t2i.hugging_face_hub import HuggingFaceHubAuthorizer
from t2i.logger import get_logger
from t2i.stable_diffusion import StableDiffusionGenerator
from t2i.text_to_image_item import TextToImageItem


class TextToImageController:
    def __init__(self):
        self.logger = get_logger(__name__)

    def preload_model(self, model: str):
        self.logger.info(f"Loading model: {model}")
        authorizer = HuggingFaceHubAuthorizer(token=env.hugging_face_hub_token)
        generator = StableDiffusionGenerator(authorizer=authorizer)
        generator.preload_model(model)
        self.logger.info(f"Loaded model: {model}")

    def create_image(self, item: TextToImageItem) -> PathLike:
        self.logger.info(f"Generating image: {item}")

        authorizer = HuggingFaceHubAuthorizer(token=env.hugging_face_hub_token)
        generator = StableDiffusionGenerator(authorizer=authorizer)
        usecase = GenerateImageUsecase(generator=generator)
        path = usecase.generate(item)
        self.logger.info(f"Generated image: {path}")
        return path
