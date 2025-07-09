import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain import hub
from doc_loader import pdf_loader, docx_loader, txt_loader
import tiktoken
from langchain_community.tools.tavily_search import TavilySearchResults


# Load .env and Gemini API
load_dotenv()

# Initialize chat history (mock DB)
chats = {
    1: [HumanMessage(content="Translate from English to French: I love programming."),
        AIMessage(content="J'adore la programmation.")]
}

# Memory for chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Token tracker for session
total_tokens = 50000
TOKEN_LIMIT = 100000

# Token counting function
def count_tokens(text, model="gpt-4-1106-preview"): # gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# File loader and retriever
path = input("Enter file path: ").strip()
extension = path.split('.')[-1].lower()

if extension == "pdf":
    vector_store = pdf_loader(path)
elif extension == "docx":
    vector_store = docx_loader(path)
elif extension == "txt":
    vector_store = txt_loader(path)
else:
    vector_store = None

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}) if vector_store else None

# AI agent function with RAG + Search + Token Tracking
def qna_agent_response(query: str, current_total: int):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    search_tool = DuckDuckGoSearchRun()
    web_search = Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool to look up information on the web ONLY when needed."
    )

    
    # # Tavily Web Search Tool
    # search_tool = TavilySearchResults()
    # web_search = Tool(
    #     name="TavilySearch",
    #     func=search_tool.run,
    #     description="Useful for answering questions about current events or factual topics from the web."
    # )

    prompt = hub.pull("hwchase17/react-chat")

    agent = create_react_agent(llm=llm, tools=[web_search], prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[web_search],
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    original_query = query

    if retriever:
        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])
        query = f"Answer based on the following context:\n{context}\n\nQuery: {original_query}"

    # Count input tokens
    input_tokens = count_tokens(query)

    # Run the agent
    result = agent_executor.invoke({'input': query})
    output = result["output"]

    # Count output tokens
    output_tokens = count_tokens(output)
    session_tokens = input_tokens + output_tokens

    print(f"Input Tokens: {input_tokens} | Output Tokens: {output_tokens} | This Turn: {session_tokens}")
    return result, session_tokens

# Chat Loop
while True:
    query = input("\n Enter your query (or 'exit' to quit): ")
    if query.lower() in ["exit", "quit", "thank you", "bye"]:
        chats[max(chats.keys()) + 1] = memory.chat_memory.messages
        print("\n Goodbye! Your chat has been saved.\n")
        print("Session Memory:\n", chats)
        break

    if total_tokens >= TOKEN_LIMIT:
        print(f" Token limit reached ({TOKEN_LIMIT} tokens). Please wait or start a new session.")
        break

    try:
        result, used_tokens = qna_agent_response(query, total_tokens)
        total_tokens += used_tokens
        print(f"\n Total Tokens Used This Session: {total_tokens}")
        print("", result["output"])
    except Exception as e:
        print("Error:", e)
