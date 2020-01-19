FROM python:3

WORKDIR /analyzer

COPY requirements.txt /analyzer
RUN pip install --no-cache-dir -r requirements.txt

COPY analyzer/ /analyzer
RUN chmod +x /analyzer/run.sh