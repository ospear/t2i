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

        # Common optimization parameters for memory efficiency
        common_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,  # Reduce CPU memory during loading
        }

        if model == SDXL1:
            if os.path.exists(model_path):
                return DiffusionPipeline.from_pretrained(
                    model_path,
                    use_safetensors=True,
                    **common_kwargs
                )

            self.authorizer.login()
            pipe = DiffusionPipeline.from_pretrained(
                model,
                use_safetensors=True,
                **common_kwargs
            )
            pipe.save_pretrained(model_path)
            return pipe

        elif model == SD3_MEDIUM:
            if os.path.exists(model_path):
                return StableDiffusion3Pipeline.from_pretrained(
                    model_path,
                    **common_kwargs
                )

            self.authorizer.login()
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                **common_kwargs
            )
            pipe.save_pretrained(model_path)
            return pipe

        raise ValueError(f"Unknown model {model}")

    def generate(self, item: TextToImageItem):
        # Clear CUDA cache and set memory allocation configuration
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Load model with CPU offloading
        pipe = self.preload_model(model=item.model)

        # Enable multiple memory optimization strategies
        pipe.enable_attention_slicing(1)  # Maximum slicing
        pipe.enable_vae_slicing()  # VAE slicing for memory efficiency

        # Enable CPU offloading for model components
        if hasattr(pipe, 'enable_model_cpu_offload'):
            pipe.enable_model_cpu_offload()
        else:
            # Fallback to sequential CPU offload
            pipe.enable_sequential_cpu_offload()

        # Clear cache before inference
        torch.cuda.empty_cache()

        # Generate image with reduced memory footprint
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

        # Cleanup
        del image, pipe
        torch.cuda.empty_cache()

        # Force garbage collection
        import gc
        gc.collect()

        return path
