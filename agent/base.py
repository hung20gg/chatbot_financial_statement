from .const import Config
class BaseAgent:
    def __init__(self, config : Config):
        self.config = config

    def get_response(self, user_input):
        return {
            'text' : 'Hello World!',
            'table' : None
        }