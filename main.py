from t2i.controller import TextToImageController
from t2i.logger import setup_logger
from t2i.text_to_image_item import TextToImageItem, SDXL1, SD3_MEDIUM


class Commands:
    def t2i(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        model: str = SDXL1,
    ):
        item = TextToImageItem(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            model=model,
        )
        path = TextToImageController().create_image(item)
        return path

    def preload_models(self):
        c = TextToImageController()
        # 6.5GB
        c.preload_model(model=SDXL1)
        # 17GB
        # c.preload_model(model=SD3_MEDIUM)


if __name__ == "__main__":
    setup_logger()

    import fire

    fire.Fire(Commands())
