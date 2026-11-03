# Container recipe for the local service. Not yet tested in Docker.
FROM python:3.12-slim
WORKDIR /app
RUN python -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-portfolio.txt ./
RUN python -m pip install --no-cache-dir -r requirements-portfolio.txt
RUN useradd --uid 10001 --create-home appuser
COPY --chown=10001:10001 . /app
ENV PYTHONPATH=/app/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false QUALIAFORM_ENCODER=/models/encoder
USER appuser
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3)"
CMD ["python", "-m", "uvicorn", "qualiaform_part2.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765"]
