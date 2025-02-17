class BaseSemantic:
    def __init__(self, **kwargs):
        pass
    
    def ensure_database_and_collections(self, db_name: str, *collections):
        pass

    def switch_collection(self, collection_name: str):
        pass
    

    def add_solver_output(self, output: dict):
        pass
    
    def create_conversation(self, user_id: str):
        pass
    
    def add_message(self, conversation_id: str, messages: list[dict], sql_messages: list[dict]):
        pass
    
    def get_messages(self, conversation_id: str):
        pass

    def sql_feedback(self, solver_id: int, output_id: int, feedback):
        pass