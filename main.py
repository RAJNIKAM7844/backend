import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import docx
import logging
from typing import List

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://10.0.2.2:8000",
    "http://192.168.56.1:8000",  # Added your computer's IP
    "http://192.168.46.146",  # Added without port just in case
    "http://192.168.56.1"
    # You can keep other IPs if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class ChatRequest(BaseModel):
    user_input: str
    chat_history: list
    
class DocumentExplainRequest(BaseModel):
    content: str

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
)

# Load FAISS index with dangerous deserialization enabled
try:
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded successfully")
except Exception as e:
    logger.error(f"Error loading FAISS index: {e}")
    raise

# Create a retriever for similarity search
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the prompt template for the chatbot
prompt_template = """<s>[INST]This is a chat template. As a legal chat bot specializing in Indian Penal Code queries, 
your primary objective is to provide accurate and concise information based on the user's questions. 
You will adhere strictly to the instructions provided, offering relevant context from the knowledge base 
while avoiding unnecessary details. Your responses will be brief and to the point.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
HUMAN: {question}
ASSISTANT:
</s>[INST]
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'question', 'chat_history']
)

# Get Together API key from environment or use a placeholder for testing
TOGETHER_AI_API = os.getenv('TOGETHER_AI', '19658ef73a2e955d391636c484aaf3f571b027ea7d7744ec97b87535c41df027')

# Initialize the LLM with Together API
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=TOGETHER_AI_API
)

# Initialize the conversational retrieval chain
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db_retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt': prompt}
)

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Create a temporary file to store the uploaded document
        temp_file_path = f"temp_{file.filename}"
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"Temporary file created: {temp_file_path}")
            
            # Extract text based on file type
            if file_extension == ".pdf":
                extracted_text = extract_text_from_pdf(temp_file_path)
                logger.info("PDF text extracted successfully")
            elif file_extension in [".docx", ".doc"]:
                extracted_text = extract_text_from_docx(temp_file_path)
                logger.info("DOCX text extracted successfully")
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            return {"message": "Document uploaded successfully", "content": extracted_text}
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Temporary file removed: {temp_file_path}")
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
@app.post("/explain-document")
async def explain_document(request: DocumentExplainRequest):
    try:
        logger.info("Received request to explain document")
        
        # Prepare the input for the QA chain
        explanation_prompt = """
        Please provide a comprehensive explanation of the following document. Include:
        1. A brief summary of the main content
        2. Key points or main ideas presented
        3. Any notable sections or structure of the document
        4. The apparent purpose or intent of the document
        5. Any legal implications or relevance, especially in the context of Indian law

        Document content: {content}
        """
        
        input_data = {
            "question": explanation_prompt.format(content=request.content),
            "chat_history": ""
        }

        # Use the existing QA chain to generate an explanation
        result = qa(input_data)

        logger.info("Document explanation generated successfully")

        return {"explanation": result["answer"]}
    except Exception as e:
        logger.error(f"Error explaining document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error explaining document: {str(e)}")
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received user input: {request.user_input}")
        logger.info(f"Received chat history: {request.chat_history}")

        # Format chat history for the prompt
        formatted_chat_history = "\n".join(
            [f"{msg['sender']}: {msg['message']}" for msg in request.chat_history]
        )

        # Prepare the input data for the QA chain
        input_data = {
            "question": request.user_input,
            "chat_history": formatted_chat_history
        }

        # Call the QA chain with the correctly structured input
        result = qa(input_data)

        logger.info(f"Response from bot: {result['answer']}")

        return {"assistant_response": result["answer"]}
    
    except Exception as e:
        logger.error(f"Error in chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")