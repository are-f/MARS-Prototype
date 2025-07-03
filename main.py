from fastapi import FastAPI
from dotenv import load_dotenv
from qna_agent import qna_agent_response  # Absolute import

load_dotenv()


app = FastAPI(title="qna Agent")

@app.get("/qna_Agent")
def QnA_Agent(input: str) -> dict:
    response = qna_agent_response(input)
    return {"response": response}

