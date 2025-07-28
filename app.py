# app.py
import feedparser
import torch
from flask import Flask, request, jsonify
from newspaper import Article, Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
import os

# --- Cấu hình logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)

# --- Cấu hình cho newspaper3k ---
# Cần có user-agent để giả lập trình duyệt, tránh bị chặn
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 20  # Tăng thời gian chờ

# --- Tải mô hình Embedding ---
# Sử dụng một mô hình đa ngôn ngữ tốt từ sentence-transformers.
# Quá trình này sẽ mất một chút thời gian ở lần khởi động đầu tiên để tải model.
# Sử dụng cache để lưu model đã tải
cache_folder = os.getenv('TRANSFORMERS_CACHE', './model_cache')
logging.info(f"Đang tải mô hình Sentence Transformer... (Cache folder: {cache_folder})")
# Kiểm tra GPU có sẵn không
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device, cache_folder=cache_folder)
logging.info(f"Đã tải xong mô hình trên thiết bị: {model.device}")


def get_full_article_text(url):
    """
    Sử dụng newspaper3k để tải và trích xuất nội dung đầy đủ của một bài báo từ URL.
    """
    try:
        article = Article(url, config=config, language='vi')
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.error(f"Lỗi khi cào dữ liệu từ URL '{url}': {e}")
        return None

def create_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Sử dụng LangChain để chia văn bản thành các chunk nhỏ hơn.
    """
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@app.route('/process', methods=['POST'])
def process_rss_feeds():
    """
    Endpoint chính để xử lý danh sách các RSS feed.
    """
    # Lấy danh sách URL từ body của request
    data = request.get_json()
    if not data or 'urls' not in data:
        return jsonify({"error": "Vui lòng cung cấp danh sách URL trong body request dưới key 'urls'."}), 400

    rss_urls = data['urls']
    all_processed_articles = []
    max_articles_per_feed = 5 # Giới hạn số bài viết mỗi feed để tránh request quá dài

    for rss_url in rss_urls:
        logging.info(f"Đang xử lý RSS feed: {rss_url}")
        feed = feedparser.parse(rss_url)

        if feed.bozo:
            logging.warning(f"Lỗi phân tích cú pháp RSS feed: {rss_url} - {feed.bozo_exception}")
            continue

        for entry in feed.entries[:max_articles_per_feed]:
            article_url = entry.get('link')
            article_title = entry.get('title')

            if not article_url:
                continue

            logging.info(f"Đang lấy nội dung bài viết: '{article_title}'")
            full_text = get_full_article_text(article_url)

            if not full_text:
                logging.warning(f"Bỏ qua bài viết '{article_title}' vì không lấy được nội dung.")
                continue

            # 2. Tạo chunks
            text_chunks = create_text_chunks(full_text)

            # 3. Tạo vectors cho từng chunk
            # API sẽ trả về cả chunk và vector tương ứng
            # Lưu ý: Trả về vector trong JSON có thể làm response rất lớn.
            # Trong ứng dụng thực tế, bạn nên lưu vector vào DB và chỉ trả về ID.
            logging.info(f"Đang tạo vector cho {len(text_chunks)} chunks...")
            chunk_vectors = model.encode(text_chunks).tolist() # tolist() để chuyển thành list có thể JSON hóa

            processed_article = {
                "title": article_title,
                "url": article_url,
                "chunks": text_chunks,
                "vectors": chunk_vectors # Thêm vector vào kết quả
            }
            all_processed_articles.append(processed_article)

    logging.info(f"Hoàn thành xử lý. Tổng cộng {len(all_processed_articles)} bài viết đã được xử lý.")
    return jsonify(all_processed_articles)

@app.route('/')
def health_check():
    """
    Endpoint kiểm tra sức khỏe của API.
    """
    return "API xử lý tin tức đang hoạt động!"

# Chạy app (chỉ dùng cho local, trên Render sẽ dùng Gunicorn)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
