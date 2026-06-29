ARG PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG APT_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN set -eux; \
    rm -f /etc/apt/sources.list.d/debian.sources; \
    codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"; \
    { \
      echo "deb http://${APT_MIRROR}/debian ${codename} main contrib non-free non-free-firmware"; \
      echo "deb http://${APT_MIRROR}/debian ${codename}-updates main contrib non-free non-free-firmware"; \
      echo "deb http://${APT_MIRROR}/debian-security ${codename}-security main contrib non-free non-free-firmware"; \
    } > /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends libmagic1; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -i ${PIP_INDEX_URL} --extra-index-url https://pypi.org/simple -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
