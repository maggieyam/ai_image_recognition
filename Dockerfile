FROM python:3.12-slim

# Hugging Face Spaces runs the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only torch keeps the image small. For a GPU Space, set the TORCH_INDEX variable
# in the Space settings to a CUDA wheel index, e.g. https://download.pytorch.org/whl/cu124,
# and rebuild; the app then uses the GPU automatically.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py recognizer.py ./
COPY --chown=user web web

# Bake model weights into the image so the first request isn't a multi-minute download
RUN python -c "from recognizer import ImageRecognizer; ImageRecognizer()"

EXPOSE 7860
# One worker: each worker loads its own copy of the model (~2GB RAM), and the
# rate limiter's counters live in the worker's memory.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120"]
