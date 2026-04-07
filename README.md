# Rasa Chatbot Project

Project nay su dung Rasa + Custom Action (RAG voi ChromaDB, Sentence Transformers, Gemini API).

## 1) Yeu cau moi truong

- Windows, macOS hoac Linux
- Python 3.10 (khuyen nghi)
- pip

> Luu y: file `actions/actions.py` dang hard-code API key Gemini. Nen doi sang bien moi truong truoc khi deploy.

## 2) Cai dat

Mo terminal tai thu muc goc project (`rasa_prj`) va chay:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu ban gap loi version voi Rasa, hay dung Python 3.10 va cai dat lai trong moi truong ao moi.

## 3) Train model

```powershell
rasa train
```

## 4) Chay bot (can 3 terminal)

### Terminal 1: chay Action Server

```powershell
.\.venv\Scripts\Activate.ps1
rasa run actions
```

Action server se lang nghe tai cong 5055 theo cau hinh trong `endpoints.yml`.

### Terminal 2: Kết nối Rasa qua REST API

```powershell
.\.venv\Scripts\Activate.ps1
rasa run --enable-api
```
### Terminal 3: Chạy UI với Streamlit

```powershell
.\.venv\Scripts\Activate.ps1
Streamlit streamlit_app.py
```

Sau do ban co the chat truc tiep trong terminal.

## 5) Chay qua REST API (tuy chon)

```powershell
.\.venv\Scripts\Activate.ps1
rasa run --enable-api --cors "*"
```

Vi du goi webhook:

```http
POST http://localhost:5005/webhooks/rest/webhook
Content-Type: application/json

{
  "sender": "user1",
  "message": "xin chao"
}
```

## 6) Kiem tra nhanh

- Dam bao thu muc `chroma_db/` ton tai (project dang dung ChromaDB persistent)
- Dam bao action server dang chay truoc khi hoi cac intent can RAG
- Thu cac intent co san: `greet`, `goodbye`, `ask_knowledge`

## 7) Cau truc chinh

- `data/`: du lieu NLU, stories, rules
- `domain.yml`: intents, responses, actions
- `config.yml`: pipeline va policies
- `endpoints.yml`: endpoint custom action
- `actions/actions.py`: logic RAG (ChromaDB + rerank + Gemini)

## 8) Lenh huu ich

```powershell
rasa data validate
rasa test
```

Neu can reset model cu va train lai:

```powershell
rasa train --force
```
