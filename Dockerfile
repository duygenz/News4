# Dockerfile
# Sử dụng một ảnh Python chính thức làm nền
FROM python:3.11-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Cài đặt các gói phụ thuộc hệ thống (nếu cần)
# newspaper3k cần một vài gói để hoạt động chính xác
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/lists/*

# Sao chép file requirements.txt vào container trước
COPY requirements.txt requirements.txt

# Tạo thư mục cache cho models và cấp quyền
RUN mkdir -p /app/model_cache && chmod -R 755 /app/model_cache

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của bạn vào container
COPY . .

# Thiết lập biến môi trường để cache model vào thư mục đã tạo
ENV TRANSFORMERS_CACHE=/app/model_cache

# Expose cổng mà ứng dụng của bạn sẽ chạy
EXPOSE 10000

# Lệnh để chạy ứng dụng khi container khởi động
# Sử dụng Gunicorn để chạy ứng dụng Flask
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "120", "app:app"]
