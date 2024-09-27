import huggingface_hub


class HuggingFaceHubAuthorizer:
    def __init__(self, token: str):
        self.token = token

    def login(self):
        huggingface_hub.login(token=self.token)
