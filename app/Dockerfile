FROM python:3.10-slim

# 安裝系統依賴（用於 PDF/DOCX 解析及健康檢查）
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝依賴套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式程式碼
COPY . .

# 建立資料目錄
RUN mkdir -p /app/data

# 啟動 FastAPI 伺服器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
