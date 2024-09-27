import os.path
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
from uuid6 import uuid7
from t2i.text_to_image_item import TextToImageItem, SDXL1, SD3_MEDIUM
from t2i.hugging_face_hub import HuggingFaceHubAuthorizer


class StableDiffusionGenerator:
    def __init__(self, authorizer: HuggingFaceHubAuthorizer):
        self.authorizer = authorizer

    def preload_model(self, model: str = SDXL1):
        model_path = str(
            Path(
                os.path.join(os.path.dirname(__file__)), "../pretrained", model
            ).resolve()
        )

        if model == SDXL1:
            if os.path.exists(model_path):
                return DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )

            self.authorizer.login()
            pipe = DiffusionPipeline.from_pretrained(
                model,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            pipe.save_pretrained(model_path)
            return pipe

        elif model == SD3_MEDIUM:
            if os.path.exists(model_path):
                return StableDiffusion3Pipeline.from_pretrained(model_path)

            self.authorizer.login()
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch.float16,
            )
            pipe.save_pretrained(model_path)
            return pipe

        raise ValueError(f"Unknown model {model}")

    def generate(self, item: TextToImageItem):
        torch.cuda.empty_cache()

        pipe = self.preload_model(model=item.model)
        pipe = pipe.to("cuda")
        image = pipe(
            item.prompt,
            negative_prompt=item.negative_prompt,
            num_inference_steps=item.num_inference_steps,
            guidance_scale=item.guidance_scale,
        ).images[0]

        path = Path(
            os.path.join(os.path.dirname(__file__), f"../tmp/{uuid7()}.png")
        ).resolve()
        image.save(path)

        del image, pipe
        torch.cuda.empty_cache()

        return path
