FROM python:3.10-slim

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y build-essential

# Çalışma dizini oluştur
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları ve model dosyalarını kopyala
COPY . .

# 8000 portunu aç
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]