# Smart Airport Passenger Assistance Chatbot - optional containerisation
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

# Build datasets, train models and run the evaluation once at build time
RUN python src/generate_images.py && python src/generate_queries.py \
    && cd src && python vision.py && python nlp.py && python evaluate.py

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.headless", "true", "--server.port", "8501"]
