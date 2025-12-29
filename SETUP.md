# Guide d'Installation - Behavioral Detection System

## Prérequis

- **Python 3.9+** (recommandé: Python 3.11)
- **Git** (pour cloner le projet)
- **Docker** (optionnel, pour les containers)

---

## Installation Rapide (Windows)

### Étape 1: Cloner le projet
```bash
git clone <repository_url>
cd behavioral_detection
```

### Étape 2: Créer l'environnement virtuel
```bash
python -m venv .venv
```

### Étape 3: Activer l'environnement virtuel
```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

### Étape 4: Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 5: Installer le projet en mode développement
```bash
pip install -e .
```

### Étape 6: Lancer l'application
```bash
# Option 1: Utiliser le script de lancement
run.bat dashboard

# Option 2: Lancer directement Streamlit
.venv\Scripts\streamlit.exe run src\interface\streamlit_app.py
```

---

## Installation Détaillée

### 1. Vérifier Python
```bash
python --version
# Doit afficher Python 3.9 ou supérieur
```

Si Python n'est pas installé, téléchargez-le depuis: https://www.python.org/downloads/

### 2. Cloner le Projet
```bash
git clone <repository_url>
cd behavioral_detection
```

### 3. Créer l'Environnement Virtuel
```bash
python -m venv .venv
```

### 4. Activer l'Environnement

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/MacOS:**
```bash
source .venv/bin/activate
```

### 5. Installer les Dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Installer le Projet
```bash
pip install -e .
```

---

## Utilisation

### Lancer le Dashboard (Interface Web)
```bash
# Windows
run.bat dashboard

# Ou directement
streamlit run src/interface/streamlit_app.py
```

Accédez à: **http://localhost:8501**

### Lancer le Détecteur Temps Réel
```bash
# Windows
run.bat detector

# Ou avec Python
python -m src.detector.realtime_detector
```

### Lancer Tous les Composants
```bash
# Windows
run.bat all
```

### Générer des Données d'Entraînement
```bash
# Windows
run.bat generator

# Ou avec Python
python -m src.generator.dataset_generator --benign 1000 --malicious 800
```

### Entraîner les Modèles
```bash
# Windows
run.bat trainer

# Ou avec Python
python -m src.models.train_models
```

### Lancer les Tests
```bash
# Windows
run.bat tests

# Ou avec pytest
pytest tests/ -v
```

---

## Installation avec Docker (Optionnel)

### 1. Installer Docker Desktop
Téléchargez depuis: https://www.docker.com/products/docker-desktop

### 2. Construire les Images
```bash
cd docker
docker-compose build
```

## 🐳 Docker (Minimalist)

Pour lancer le projet via Docker (Recommended / Recommandé):

```bash
# Construire et lancer | Build and Run
docker-compose up --build

# Lancer en arrière-plan | Run in background
docker-compose up -d --build

# Arrêter | Stop
docker-compose down
```

L'application sera accessible sur: http://localhost:8501

### 4. Accéder à l'Application
- Dashboard: http://localhost:8501

---

## Structure du Projet

```
behavioral_detection/
├── config/               # Fichiers de configuration
├── data/                 # Données (générées/entraînement)
├── docker/               # Fichiers Docker
├── logs/                 # Fichiers de logs
├── src/
│   ├── collector/        # Collecte des événements système
│   ├── detector/         # Détection en temps réel
│   ├── features/         # Extraction des features
│   ├── generator/        # Génération de données
│   ├── interface/        # Interface utilisateur (Streamlit)
│   └── models/           # Modèles ML
├── tests/                # Tests unitaires
├── requirements.txt      # Dépendances Python
├── run.bat              # Script de lancement (Windows)
├── run.py               # Script de lancement (Python)
└── setup.py             # Configuration du package
```

---

## Modèles ML Disponibles

| Modèle | Type | Description |
|--------|------|-------------|
| Isolation Forest | Anomaly Detection | Détection d'anomalies non supervisée |
| Random Forest | Classification | Forêt aléatoire supervisée |
| XGBoost | Classification | Gradient boosting optimisé |
| One-Class SVM | Anomaly Detection | SVM à une classe |
| LOF | Anomaly Detection | Local Outlier Factor |

---

## Dépannage

### Erreur: "streamlit not found"
```bash
pip install streamlit
```

### Erreur: "Module not found"
Assurez-vous d'avoir activé l'environnement virtuel:
```bash
.venv\Scripts\activate.bat
```

### Erreur de port (8501 déjà utilisé)
```bash
streamlit run src/interface/streamlit_app.py --server.port=8502
```

### Arrêter les processus en cours
```bash
taskkill /F /IM streamlit.exe
taskkill /F /IM python.exe
```

---

## Support

Pour toute question ou problème, ouvrez une issue sur le dépôt GitHub.
