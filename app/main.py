"""
FastAPI RAG 服務 - 整合 Ollama + Qdrant + LangChain
功能：
1. 健康檢查
2. 聊天（支援 RAG 檢索）
3. 文件上傳與向量化
4. 列出可用模型

修正重點：
1. 環境變數名稱與 docker-compose 一致（LLM_MODEL / EMBEDDING_MODEL）
2. temperature 參數正確傳遞
3. RAG 無結果時給予使用者提示
4. 修正 PDF/DOCX 解析錯誤處理
"""

import os
import logging
import tempfile
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import requests

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# 設定 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境變數 - 修正為與 docker-compose 一致的名稱
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")  # 修正：LLM_MODEL
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")  # 修正：EMBEDDING_MODEL
COLLECTION_NAME = "documents"

# FastAPI 實例
app = FastAPI(
    title="RAG API Service",
    description="整合 Ollama + Qdrant 的檢索增強生成服務",
    version="2.0.0"
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────── Pydantic 模型 ───────────

class ChatRequest(BaseModel):
    """聊天請求格式"""
    message: str = Field(..., min_length=1, max_length=4000, description="使用者訊息")
    model: Optional[str] = Field(None, description="模型名稱，預設使用 DEFAULT_MODEL")
    use_rag: bool = Field(True, description="是否啟用 RAG 檢索")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="溫度參數")

class ChatResponse(BaseModel):
    """聊天回應格式"""
    response: str = Field(..., description="AI 回應內容")
    model: str = Field(..., description="使用的模型")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="RAG 檢索到的來源文件")

class ModelInfo(BaseModel):
    """模型資訊"""
    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None

class UploadResponse(BaseModel):
    """上傳回應格式"""
    status: str = "success"
    filename: str
    chunk_count: int
    message: str

# ─────────── 初始化客戶端 ───────────

def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.7) -> ChatOllama:
    """取得 Ollama Chat 模型 - 修正：接受 temperature 參數"""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=model,
        temperature=temperature,
        num_predict=2048,
    )

def get_embeddings() -> OllamaEmbeddings:
    """取得 Ollama Embedding 模型"""
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBED_MODEL,
    )

def get_qdrant_client() -> QdrantClient:
    """取得 Qdrant 客戶端"""
    return QdrantClient(url=QDRANT_URL)

def get_vector_store() -> Optional[QdrantVectorStore]:
    """取得向量儲存，若 collection 不存在則回傳 None"""
    try:
        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=get_embeddings(),
        )
        # 測試連線
        client.get_collection(COLLECTION_NAME)
        return vector_store
    except Exception as e:
        logger.warning(f"向量儲存尚未初始化: {e}")
        return None

# ─────────── API 端點 ───────────

@app.get("/", tags=["系統"])
async def root():
    """根端點，顯示服務資訊"""
    return {
        "service": "RAG API Service",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["系統"])
async def health_check():
    """健康檢查：確認 Ollama 與 Qdrant 連線狀態"""
    status = {
        "status": "healthy",
        "services": {}
    }
    
    # 檢查 Ollama
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            status["services"]["ollama"] = {
                "status": "connected",
                "models": [m["name"] for m in models]
            }
        else:
            status["services"]["ollama"] = {"status": "error", "message": f"HTTP {resp.status_code}"}
    except Exception as e:
        status["services"]["ollama"] = {"status": "disconnected", "message": str(e)}
    
    # 檢查 Qdrant
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        status["services"]["qdrant"] = {
            "status": "connected",
            "collections": [c.name for c in collections]
        }
    except Exception as e:
        status["services"]["qdrant"] = {"status": "disconnected", "message": str(e)}
    
    # 整體狀態
    if status["services"].get("ollama", {}).get("status") != "connected":
        status["status"] = "degraded"
    
    return status

@app.get("/models", tags=["系統"])
async def list_models():
    """列出 Ollama 中可用的模型"""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return {
                "models": [
                    ModelInfo(
                        name=m["name"],
                        size=m.get("size"),
                        modified_at=m.get("modified_at")
                    ) for m in models
                ]
            }
        raise HTTPException(status_code=500, detail="Ollama API 回應異常")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="無法連線到 Ollama 服務")

@app.post("/chat", response_model=ChatResponse, tags=["聊天"])
async def chat(request: ChatRequest):
    """
    聊天端點，支援 RAG 檢索增強
    
    流程：
    1. 若啟用 RAG，先檢索相關文件
    2. 將檢索結果與使用者訊息組合成 prompt
    3. 呼叫 Ollama 生成回應
    """
    model = request.model or DEFAULT_MODEL
    
    # 取得 LLM - 修正：傳遞 temperature
    try:
        llm = get_llm(model, request.temperature)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"無法初始化模型 {model}: {str(e)}")
    
    sources = []
    context_text = ""
    
    # RAG 檢索
    if request.use_rag:
        vector_store = get_vector_store()
        if vector_store:
            try:
                docs = vector_store.similarity_search(request.message, k=3)
                if docs:
                    context_text = "\n\n".join([
                        f"【文件來源 {i+1}】\n{doc.page_content}" 
                        for i, doc in enumerate(docs)
                    ])
                    sources = [
                        {
                            "index": i + 1,
                            "content": doc.page_content[:200],
                            "metadata": doc.metadata
                        }
                        for i, doc in enumerate(docs)
                    ]
                else:
                    # 修正：無檢索結果時提示
                    sources = [{"info": "未檢索到相關文件"}]
            except Exception as e:
                logger.warning(f"RAG 檢索失敗: {e}")
                sources = [{"info": f"RAG 檢索失敗: {str(e)}"}]
        else:
            sources = [{"info": "向量資料庫尚未初始化，請先上傳文件"}]
    
    # 建構 prompt
    if context_text:
        system_prompt = (
            "你是一個專業的知識助手，請根據以下檢索到的文件內容回答使用者問題。\n"
            "如果文件內容不足以回答，請明確告知。\n\n"
            f"【檢索文件】\n{context_text}\n\n"
            f"【使用者問題】\n{request.message}"
        )
    else:
        # 無 RAG 結果時，直接以一般模式回答
        system_prompt = request.message
    
    # 呼叫 LLM
    try:
        response = llm.invoke(system_prompt)
        return ChatResponse(
            response=response.content,
            model=model,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型生成失敗: {str(e)}")

@app.post("/upload", response_model=UploadResponse, tags=["文件管理"])
async def upload_file(file: UploadFile = File(...)):
    """
    上傳文件並向量化儲存到 Qdrant
    
    支援格式：txt, md, pdf, docx
    流程：
    1. 讀取文件內容
    2. 切割為 chunks
    3. 建立 Embedding 並存入 Qdrant
    """
    # 檢查檔案類型
    allowed_extensions = {".txt", ".md", ".pdf", ".docx"}
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支援的檔案類型: {ext}，支援: {allowed_extensions}")
    
    # 讀取檔案內容
    content = await file.read()
    
    # 使用 TemporaryDirectory 確保臨時檔案自動清理（避免 Windows 鎖檔）
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 根據檔案類型解析內容
            if ext == ".txt" or ext == ".md":
                text_content = content.decode("utf-8", errors="ignore")
            elif ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                tmp_path = os.path.join(temp_dir, "temp.pdf")
                with open(tmp_path, "wb") as tmp:
                    tmp.write(content)
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_content = "\n".join([d.page_content for d in docs])
            elif ext == ".docx":
                from langchain_community.document_loaders import Docx2txtLoader
                tmp_path = os.path.join(temp_dir, "temp.docx")
                with open(tmp_path, "wb") as tmp:
                    tmp.write(content)
                loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
                text_content = "\n".join([d.page_content for d in docs])
            else:
                text_content = content.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"檔案解析失敗: {str(e)}")
    
    # 切割文字 - 調整 chunk_size 為 500 以提升檢索精準度
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text_content)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="檔案內容為空，無法向量化")
    
    # 建立 Document 列表
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": filename, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # 存入 Qdrant
    try:
        vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=get_embeddings(),
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            force_recreate=False,  # 保留既有資料
        )
        
        return UploadResponse(
            status="success",
            filename=filename,
            chunk_count=len(chunks),
            message=f"成功向量化 {len(chunks)} 個文字區塊"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量化儲存失敗: {str(e)}")

# ─────────── 主程式入口 ───────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
