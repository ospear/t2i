from pydantic import BaseModel, Field


SD3_MEDIUM = "stabilityai/stable-diffusion-3-medium-diffusers"
SDXL1 = "stabilityai/stable-diffusion-xl-base-1.0"


class TextToImageItem(BaseModel):
    prompt: str = Field("", description="prompt")
    negative_prompt: str = Field("", description="negative prompt")
    num_inference_steps: int = Field(30, description="num inference steps")
    guidance_scale: float = Field(7.0, description="guidance scale")
    model: str = Field(SDXL1, description="model name")
