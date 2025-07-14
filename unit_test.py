import pytest
from unittest.mock import MagicMock
from generator import get_query_type
from preprocess import correct_accent, clean_text, remove_urls, remove_emails
from retriever import get_best_files
from redis_db import RedisChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Test de get_query_type
def test_get_query_type():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    mock_model.return_value.logits = [[1.0, 2.0]]  # Mocked logits without TensorFlow
    
    result = get_query_type("example query", mock_model, mock_tokenizer)
    assert result in ["simple", "complex"]

# Test de correct_accent
def test_correct_accent():
    assert correct_accent("caf´e") == "café"
    assert correct_accent("`e`a`u") == "èàù"
    assert correct_accent("normal text") == "normal text"

# Test de clean_text
def test_clean_text():
    assert clean_text("Hello!!  World!!") == "Hello!! World!!"
    assert clean_text("This is a test.") == "This is a test."

# Test de remove_urls
def test_remove_urls():
    assert remove_urls("Check this out: https://example.com") == "Check this out: "
    assert remove_urls("No URL here.") == "No URL here."

# Test de remove_emails
def test_remove_emails():
    assert remove_emails("Contact me at test@example.com") == "Contact me at "
    assert remove_emails("No email here.") == "No email here."

# Test de get_best_files
def test_get_best_files():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        MagicMock(metadata={"source": "file1.pdf"}),
        MagicMock(metadata={"source": "file2.pdf"})
    ]
    result = get_best_files("test query", mock_retriever, k=2)
    assert result == ["file1.pdf", "file2.pdf"]

# Tests pour RedisChatMessageHistory
def test_redis_chat_message_history():
    mock_redis = MagicMock()
    session_id = "test_session"
    chat_history = RedisChatMessageHistory(session_id, mock_redis)
    
    # Simuler l'ajout de messages
    chat_history.add_message(HumanMessage(content="Hello"))
    chat_history.add_message(SystemMessage(content="System message"))
    chat_history.add_ai_message("AI response")
    
    assert mock_redis.rpush.call_count == 3  # Vérifie que les messages sont bien stockés
    
    # Simuler la récupération des messages
    mock_redis.lrange.return_value = [
        '{"role": "user", "content": "Hello"}',
        '{"role": "system", "content": "System message"}',
        '{"role": "assistant", "content": "AI response"}'
    ]
    messages = chat_history.get_messages()
    
    assert len(messages) == 3
    assert isinstance(messages[0], HumanMessage) and messages[0].content == "Hello"
    assert isinstance(messages[1], SystemMessage) and messages[1].content == "System message"
    assert isinstance(messages[2], AIMessage) and messages[2].content == "AI response"
