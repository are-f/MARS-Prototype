import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain import hub
from doc_loader import pdf_loader, docx_loader, txt_loader

# Load environment variables
load_dotenv()

# chats Database--------------------------------------------- Dummy data--------------------------------------
# --------------------------------------------------------FETCH FROM DATABASE--------------------------------------------
chats = {
    1: [
        HumanMessage(content="Translate from English to French: I love programming."),
        AIMessage(content="J'adore la programmation."),
        HumanMessage(content="What did you just say?"),
    ],
    2: [
        HumanMessage(content="Tell me about the Major Cities"),
        AIMessage(content="It's NewYork, Paris, Bejing, Hongkong"),
        HumanMessage(content="What did you just say?"),
    ],
    3: [HumanMessage(content='Tell me about RAG'),
        AIMessage(content='Retrieval-Augmented Generation (RAG) is an innovative approach in natural language processing (NLP) that enhances the quality of generated text by combining retrieval-based and generation-based models. It augments Large Language Models (LLMs) by adding an information retrieval system, giving control over the grounding data used when formulating a response. RAG allows for incorporating current, domain-specific data into language model-based applications. Key practices used across the RAG pipeline include full-text search, vector search, chunking, hybrid search, query rewriting, and re-ranking.'), 
        HumanMessage(content='How can I learn it'), 
        AIMessage(content='You can learn about Retrieval-Augmented Generation (RAG) through resources that explain how it combines retrieval-based and generation-based models to enhance the quality of generated text. Look for information on how RAG systems access and incorporate external knowledge sources in real-time to provide accurate and contextually relevant information. Several articles and guides online cover the workings, applications, benefits, and best practices of RAG. Also, consider exploring how RAG patterns are used in platforms like Azure AI Search solutions.'), 
        HumanMessage(content='okay anything else you want to recommend'), 
        AIMessage(content="To further your understanding of RAG, I recommend exploring specific implementations and use cases. Look into open-source RAG frameworks like LangChain and LlamaIndex, as they provide practical examples and tools for building RAG pipelines. Also, consider investigating how RAG is being applied in different industries, such as healthcare, finance, and e-commerce, to understand its real-world impact. Finally, keep an eye on research papers and blog posts that discuss the latest advancements and challenges in RAG, as the field is constantly evolving.\n```")]

}
# ---------------------------------------------------------------------------------------

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# File path and loader
path = input("Enter File path: ")
extension = path.split('.')[-1].lower()

if extension == "pdf":
    vector_store = pdf_loader(path)
elif extension == "docx":
    vector_store = docx_loader(path)
elif extension == "txt":
    vector_store = txt_loader(path)
else:
    vector_store = None

retriever = None
if vector_store:
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# QnA Agent Function
def qna_agent_response(query: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
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
        chats[list(chats.keys())[-1] + 1] = memory.chat_memory.messages #--------------------Update in database SAVE ON BACKEND
        print("Good Bye!")
        print(chats)
        break
    try:
        result = qna_agent_response(query)
        print(result["output"])
    except Exception as e:
        print("Error:", e)
