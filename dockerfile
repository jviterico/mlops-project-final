FROM python:3.10

WORKDIR /app1

COPY app1.py .
COPY models/ ./models/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app1:app", "--host", "0.0.0.0", "--port", "80"]

