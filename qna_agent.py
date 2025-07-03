import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain import hub
from doc_loader import pdf_loader, docx_loader, load_json_texts, txt_loader, csv_docs_loader

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# File path and loader
path = input("Enter File path: ")
extension = path.split('.')[-1].lower()

if extension == "pdf":
    vector_store = pdf_loader(path)
elif extension == "csv":
    vector_store = csv_docs_loader(path)
elif extension == "docx":
    vector_store = docx_loader(path)
elif extension == "txt":
    vector_store = txt_loader(path)
elif extension == "json":
    vector_store = load_json_texts(path)
else:
    vector_store = None

retriever = None
if vector_store:
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# QnA Agent Function
def qna_agent_response(query: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", GOOGLE_API_KEY=GOOGLE_API_KEY)
    search_tool = DuckDuckGoSearchRun()
    tool = Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool to look up information on the web ONLY when needed."
    )

    prompt = hub.pull("hwchase17/react-chat")
    agent_tools = [tool]

    agent = create_react_agent(llm=llm, tools=agent_tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    if retriever:
        docs = retriever.invoke(query)
        # Combine context from documents
        doc_context = "\n".join([d.page_content for d in docs])
        query = f"Answer based on the following context:\n{doc_context}\n\nQuery: {query}"

    result = agent_executor.invoke({'input': query})
    return result

# Interaction Loop
while True:
    query = input("Enter your Query: ")
    if query.lower() in ["exit", "quit", "thank you", "bye"]:
        print("Good Bye!")
        break
    try:
        result = qna_agent_response(query)
        print(result["output"])
    except Exception as e:
        print("Error:", e)
