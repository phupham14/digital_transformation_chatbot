import streamlit as st
import requests
import uuid

# ==================== Config ====================
RASA_WEBHOOK = "http://localhost:5005/webhooks/rest/webhook"

# ==================== Page Setup ====================
st.set_page_config(
    page_title="Chatbot Chuyển Đổi Số",
    page_icon="🤖",
    layout="centered",
)

# ==================== Session State ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]


# ==================== Helper ====================
def send_to_rasa(message: str) -> str:
    """Gửi tin nhắn tới Rasa REST API và trả về phản hồi."""
    try:
        payload = {"sender": st.session_state.session_id, "message": message}
        response = requests.post(RASA_WEBHOOK, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if data:
                return "\n\n".join(msg.get("text", "") for msg in data if msg.get("text"))
            return "_(Bot không có phản hồi)_"
        return f"❌ Lỗi kết nối Rasa (HTTP {response.status_code})"
    except requests.exceptions.ConnectionError:
        return "❌ Không kết nối được Rasa. Hãy đảm bảo `rasa run --enable-api` đang chạy."
    except requests.exceptions.Timeout:
        return "⏱️ Rasa phản hồi quá lâu (timeout 60s). Vui lòng thử lại."
    except Exception as e:
        return f"❌ Lỗi: {e}"


# ==================== UI ====================
st.title("🤖 Chatbot Chuyển Đổi Số")
st.markdown("""
Hỏi đáp dựa trên **Cẩm nang Chuyển đổi số 2021**.  
Powered by **Rasa + ChromaDB + Google Gemini**.
""")

# Nút xoá lịch sử
if st.button("🗑️ Xoá lịch sử chat"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.rerun()

st.divider()

# Hiển thị lịch sử tin nhắn
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập liệu
if user_input := st.chat_input("Nhập câu hỏi về chuyển đổi số..."):
    # Hiển thị tin nhắn user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Gửi tới Rasa và hiển thị phản hồi
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu..."):
            bot_reply = send_to_rasa(user_input)
        st.markdown(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
