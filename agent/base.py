class BaseAgent:
    def __init__(self, config):
        self.config = config

    def get_response(self, user_input):
        raise NotImplementedError