FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt

RUN python -c "from pathlib import Path; p=Path('/tmp/requirements.txt'); p.write_text(p.read_text().replace('tensorflow==2.17.0', 'tensorflow-cpu==2.17.0'))"

RUN pip install --no-cache-dir --default-timeout=1800 --retries 10 -r /tmp/requirements.txt

COPY backend ./backend
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]