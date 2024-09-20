# Utilise une image officielle de Python comme base
FROM python:3.11

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie les fichiers de requirements dans le conteneur
COPY requirements.txt .

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste de l'application dans le conteneur
COPY . .

# Expose le port que l'application va utiliser
EXPOSE 8000

# Commande pour lancer l'application
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
