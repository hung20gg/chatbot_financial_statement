class BaseSemantic:
    def __init__(self, **kwargs):
        pass
    
    def switch_collection(self, collection_name):
        pass
    
    def add_sql(self, conversation_id, task, sql):
        pass
    
    def create_conversation(self, user_id):
        pass
    
    def add_message(self, conversation_id, messages, sql_messages):
        pass
    
    def get_messages(self, conversation_id):
        pass