from redis import Redis, ConnectionPool
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage  # Assurez-vous que AIMessage est défini

# Configuration de la connexion Redis.
# Pour la confidentialité, vous pouvez activer TLS ou utiliser des ACLs dans votre configuration Redis.

def create_redis_client():
    pool = ConnectionPool(host='127.0.0.1', port=6379, max_connections=100, decode_responses=True)
    return Redis(connection_pool=pool)


class RedisChatMessageHistory:
    """
    Classe pour gérer l’historique des messages d’une session via Redis.
    Chaque message est stocké dans une liste sous la clé "chat_history:<session_id>".
    """
    def __init__(self, session_id: str, redis_client: Redis):
        self.session_id = session_id
        self.redis_client = redis_client
        self.key = f"chat_history:{session_id}"
    
    def get_messages(self):
        """Récupère l’ensemble des messages depuis Redis."""
        messages = []
        for message_json in self.redis_client.lrange(self.key, 0, -1):
            data = json.loads(message_json)
            role = data.get("role")
            content = data.get("content")
            if role == "system":
                msg = SystemMessage(content=content)
            elif role == "user":
                msg = HumanMessage(content=content)
            elif role == "assistant":
                msg = AIMessage(content=content)
                # si on a enregistré la durée, on l’attache à l’objet
                if "duration" in data:
                    setattr(msg, "duration", data["duration"])
            else:
                msg = HumanMessage(content=content)
            messages.append(msg)
        return messages

    @property
    def messages(self):
        """Permet d’accéder aux messages comme à une propriété (pour compatibilité avec votre code existant)."""
        return self.get_messages()
    
    def add_message(self, message):
        """Ajoute un message à l’historique dans Redis."""
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "unknown"
        data = {"role": role, "content": message.content}
        # La commande RPush est atomique sur Redis.
        self.redis_client.rpush(self.key, json.dumps(data))
    
    def add_ai_message(self, content: str):
        """Méthode utilitaire pour ajouter un message de l'assistant."""
        ai_msg = AIMessage(content=content)
        self.add_message(ai_msg)

def get_chat_history(redis_cl, session_id: str) -> RedisChatMessageHistory:
    """
    Récupère (ou crée) l'historique de chat pour une session donnée via Redis.
    """
    return RedisChatMessageHistory(session_id, redis_cl)
