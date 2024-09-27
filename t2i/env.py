import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

hugging_face_hub_token = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
