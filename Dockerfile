FROM python:3.12-slim
WORKDIR /app
RUN mkdir -p /data
COPY backend/ ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt
ENV AXROPUS_DATABASE_URL=sqlite:////data/axropus.db
ENV AXROPUS_JWT_SECRET=change-me
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
