FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src

ENTRYPOINT ["python", "/app/src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]

EXPOSE 9009