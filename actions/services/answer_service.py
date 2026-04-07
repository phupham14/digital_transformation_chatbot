import google.generativeai as genai


class GeminiAnswerService:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_answer(self, query: str, context: str, chat_history: str) -> str:
        prompt = f"""Ban la tro ly AI chuyen ve chuyen doi so. Hay tra loi dua tren context duoc cung cap.

YEU CAU:
- Chi su dung thong tin trong [Context], KHONG tu y them kien thuc ben ngoai.
- Neu context khong chua thong tin lien quan, tra loi: "Toi khong tim thay thong tin phu hop trong tai lieu."
- KHONG sao chep nguyen van tu context, hay dien dat lai gon gang.
- Chi lay thong tin LIEN QUAN TRUC TIEP den cau hoi.
- Neu co nhieu y, hay tong hop thanh 3-4 y chinh.
- Khong viet doan van dai, uu tien bullet point.
- Tranh lap y, tranh thong tin thua.
- Trich dan so trang o CUOI CAU TRA LOI.

FORMAT TRA LOI:
📌 {query}

- <Y chinh 1>
- <Y chinh 2>
- <Y chinh 3>

(Nguồn: Trang X)

NGU CANH:
- Su dung [Lich su tro chuyen] de hieu ro cau hoi hien tai.

Lich su tro chuyen gan nhat:
{chat_history}

Context:
{context}

Cau hoi: {query}

Tra loi:
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as exc:
            print(f"[ERROR] Gemini API call failed: {exc}")
            return "Xin loi, da xay ra loi khi xu ly cau hoi cua ban."
