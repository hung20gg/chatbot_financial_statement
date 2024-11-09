from pydantic import BaseModel
from .const import Config
class BaseAgent(BaseModel):
    def __init__(self, config : Config):
        self.config = config

    def get_response(self, user_input):
        return {
            'text' : 'Hello World!',
            'table' : None
        }