from .base import BaseAgent

class Text2SQL(BaseAgent):
    def get_response(self, user_input):
        return "I am a text to SQL agent."