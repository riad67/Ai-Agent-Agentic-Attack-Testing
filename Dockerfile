FROM public.ecr.aws/docker/library/python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM public.ecr.aws/docker/library/node:22-alpine AS frontend
WORKDIR /ui
COPY ui/package*.json ./
RUN npm ci
COPY ui/ .
RUN npm run build

FROM runtime
COPY engine/ engine/
COPY data/raw/ data/raw/
COPY bundled/ bundled/
COPY server.py .
COPY tests/test_lab.py tests/test_lab.py
COPY --from=frontend /ui/dist ui/dist
RUN useradd --create-home lab && mkdir state && chown lab:lab state
USER lab
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
