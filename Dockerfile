# 1. On part d'un Linux 
FROM python:3.10-slim

# On installe la dépendance système requise par LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 2. On se place dans le dossier de travail
WORKDIR /code

# 3. On installe 'uv'
RUN pip install uv

# 4. On copie les dépendances
COPY pyproject.toml uv.lock ./

# 5. On installe les librairies Python
RUN uv sync --frozen --no-dev

# 6. On copie tout ton dossier 'app'
COPY app ./app

# 7. On expose le port
EXPOSE 8000

# 8. Commande de démarrage
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
