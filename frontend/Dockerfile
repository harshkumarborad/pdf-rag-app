FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit config: disable telemetry, set port
RUN mkdir -p /root/.streamlit
RUN echo '[server]\nport = 8501\nheadless = true\nenableXsrfProtection = false\n[browser]\ngatherUsageStats = false\n[theme]\nbase = "light"\nbackgroundColor = "#f8fafc"\nfont = "sans serif"' > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
