FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y -qq libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


FROM base as builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y -qq build-essential software-properties-common clang git curl && \
    apt-get clean

COPY ./pyproject.toml /app
COPY ./uv.lock /app

ENV PATH="/app/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/install.sh -o install.sh && \
    env UV_INSTALL_DIR="/app" sh install.sh && \
    uv python install 3.10 && \
    uv sync


FROM base as common
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY t2i /app/t2i
COPY main.py /app/main.py
COPY logging_config.yml /app/logging_config.yml


FROM common as development
EXPOSE 8000
WORKDIR /app
CMD ["uvicorn", "t2i.app:app", "--host=0.0.0.0", "--port=8000", "--reload", "--log-config=/app/logger_config.yml"]


FROM common as production
EXPOSE 8000
WORKDIR /app
COPY pretrained /app/pretrained
CMD ["uvicorn", "t2i.app:app", "--host=0.0.0.0", "--port=8000", "--workers=2", "--log-config=/app/logger_config.yml"]
