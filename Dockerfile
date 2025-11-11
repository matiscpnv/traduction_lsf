# Utiliser une image officielle Python compatible MediaPipe
FROM python:3.10-slim

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port Flask
EXPOSE 10000

# Lancer l’application avec gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
