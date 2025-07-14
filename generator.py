import paths
import redis_db
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
import asyncio
import time
import json
from datetime import datetime


# Initialize Redis Client
redis_client = redis_db.create_redis_client()




def get_model():
    """Creates and returns the LlamaCpp model."""
    llm_langchain = ChatLlamaCpp(
        model_path=paths.generator_model_path,
        n_ctx=50000,
        temperature=0.01,
        max_tokens=512,
        top_p=1,
        n_threads=2
    )
    return llm_langchain




def get_chain_generator(generator):
    """Creates the retrieval and response generation pipeline."""
    SYSTEM_TEMPLATE = """
        Answer the user's questions based on the below context. 
        If the context doesn't contain relevant information, just say "I don't know".

        <context>
        {context}
        </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(generator, question_answering_prompt)
    return document_chain




def get_chat_hist_instance(session_id):
    """Fetches chat history from Redis for the given session."""
    return redis_db.get_chat_history(redis_client, session_id)




async def generate_chat_st(user_input, generator_chain, chat_history_instance):
    """
    Generates chatbot responses asynchronously.
    Uses history stored in Redis for continuity.
    """

    system_prompt_content = (
        "You are a highly helpful assistant, and your name is llama_chat. "
        "Your answers will be concise and direct. "
        "If the provided context lacks relevant information, you may answer without it."
    )

    # Add system message if not already present
    if not any(isinstance(msg, SystemMessage) for msg in chat_history_instance.messages):
        system_message = SystemMessage(content=system_prompt_content)
        chat_history_instance.add_message(system_message)

    # Add user message
    user_message = HumanMessage(content=user_input)
    chat_history_instance.add_message(user_message)

    # Trim history to manage token limits
    history = trim_messages(
        chat_history_instance.messages,
        strategy="last",
        start_on="human",
        end_on=("human", "tool"),
        token_counter=len,
        include_system=True,
        max_tokens=15,
    )

    #  Ensure `ainvoke()` is awaited properly
    start = time.time()
    response_dict = await asyncio.ensure_future(generator_chain.ainvoke({"messages": history}))
    end = time.time()
    elapsed = end - start
    response = response_dict["answer"]

    # Save response in chat history
    #chat_history_instance.add_ai_message(response)

    # Persist message assistant *une seule fois* dans Redis,
    # avec le champ duration en secondes (float).
    key = chat_history_instance.key  # e.g. "chat_history:<session_id>"
    data = {
        "role": "assistant",
        "content": response,
        "duration": elapsed
    }
    chat_history_instance.redis_client.rpush(key, json.dumps(data))

    # duration store for the dashboard
    date_str = datetime.utcnow().date().isoformat()
    chat_history_instance.redis_client.rpush(
        f"response_times:{date_str}",
        elapsed
    )

    # daily user counter AI increase
    chat_history_instance.redis_client.incr(f"responses:{date_str}")


    return chat_history_instance , elapsed

