from .const import Config
from pydantic import BaseModel, ConfigDict
class BaseAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: Config

    def get_response(self, user_input):
        return {
            'text' : 'Hello World!',
            'table' : None
        }