from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
import os
from dotenv import load_dotenv
import shutil

load_dotenv()

app = FastAPI()

# Global variables
vectorstore = None
qa_chain = None

# Initialize Azure OpenAI
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZ_OPENAI_API_KEY"),
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15"
)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZ_OPENAI_API_KEY"),
    azure_deployment="gpt-35-turbo",
    openai_api_version="2023-05-15",
    temperature=0
)

class Question(BaseModel):
    question: str

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG PDF Chat - FastAPI</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 50px auto; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .section { 
            margin-bottom: 30px; 
            padding: 20px; 
            border: 1px solid #ddd; 
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        h2 { 
            color: #333;
            margin-top: 0;
        }
        input[type="file"], input[type="text"] { 
            width: 100%; 
            padding: 10px; 
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button { 
            padding: 10px 20px; 
            background: #007bff; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            font-size: 14px;
        }
        button:hover { 
            background: #0056b3; 
        }
        #chat-box { 
            height: 300px; 
            overflow-y: auto; 
            border: 1px solid #ddd; 
            padding: 10px; 
            margin: 10px 0; 
            background: #fafafa;
            border-radius: 4px;
        }
        .message { 
            margin: 10px 0; 
            padding: 10px; 
            border-radius: 5px;
            line-height: 1.5;
        }
        .user { 
            background: #e3f2fd; 
            text-align: right;
            margin-left: 50px;
        }
        .bot { 
            background: #f1f8e9;
            margin-right: 50px;
        }
        #status { 
            margin-top: 10px; 
            padding: 10px;
            border-radius: 4px;
        }
        .success {
            color: #155724;
            background-color: #d4edda;
        }
        .error {
            color: #721c24;
            background-color: #f8d7da;
        }
    </style>
</head>
<body>
    <h1>🤖 RAG PDF Chat - FastAPI</h1>
    
    <div class="section">
        <h2>📄 Step 1: Upload PDF</h2>
        <input type="file" id="pdf-file" accept=".pdf">
        <button onclick="uploadPDF()">Upload & Process PDF</button>
        <div id="status"></div>
    </div>
    
    <div class="section">
        <h2>💬 Step 2: Ask Questions</h2>
        <div id="chat-box"></div>
        <input type="text" id="question" placeholder="Type your question here..." onkeypress="if(event.key==='Enter') askQuestion()">
        <button onclick="askQuestion()">Ask Question</button>
    </div>

    <script>
        async function uploadPDF() {
            const fileInput = document.getElementById('pdf-file');
            const file = fileInput.files[0];
            const statusDiv = document.getElementById('status');
            
            if (!file) {
                statusDiv.className = 'error';
                statusDiv.innerText = '⚠️ Please select a PDF file';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            statusDiv.className = '';
            statusDiv.innerText = '⏳ Processing PDF...';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    statusDiv.className = 'success';
                    statusDiv.innerText = '✅ ' + result.message;
                } else {
                    statusDiv.className = 'error';
                    statusDiv.innerText = '❌ ' + result.message;
                }
            } catch (error) {
                statusDiv.className = 'error';
                statusDiv.innerText = '❌ Error: ' + error.message;
            }
        }
        
        async function askQuestion() {
            const question = document.getElementById('question').value.trim();
            
            if (!question) {
                alert('Please enter a question');
                return;
            }
            
            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="message user"><strong>You:</strong> ${question}</div>`;
            
            document.getElementById('question').value = '';
            
            const loadingId = 'loading-' + Date.now();
            chatBox.innerHTML += `<div class="message bot" id="${loadingId}"><strong>Bot:</strong> Thinking...</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                
                const result = await response.json();
                
                document.getElementById(loadingId).remove();
                
                chatBox.innerHTML += `<div class="message bot"><strong>Bot:</strong> ${result.answer}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (error) {
                document.getElementById(loadingId).remove();
                chatBox.innerHTML += `<div class="message bot"><strong>Error:</strong> ${error.message}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_CONTENT

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, qa_chain
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save file temporarily
    file_path = f"temp_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load PDF
        pdf_reader = PdfReader(file_path)
        
        # Extract text from all pages
        documents = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            doc = Document(page_content=text, metadata={"page": page_num + 1})
            documents.append(doc)
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        # Clean up
        os.remove(file_path)
        
        return {"message": f"PDF processed successfully! Created {len(chunks)} chunks from {len(documents)} pages."}
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: Question):
    global qa_chain
    
    if qa_chain is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first!")
    
    if not question.question:
        raise HTTPException(status_code=400, detail="Please provide a question")
    
    try:
        result = qa_chain.invoke({"query": question.question})
        return {"answer": result['result']}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)