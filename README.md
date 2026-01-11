# Système de Détection Comportementale

Un système complet de détection d'anomalies comportementales implémenté en Python, capable de distinguer en temps réel les comportements normaux des comportements suspects.

> **Avertissement** : Ce projet est à des fins éducatives uniquement. L'exécution ou le téléchargement de logiciels malveillants réels est strictement interdit.

## Description

Le Système de Détection Comportementale est conçu pour surveiller les activités du système, collecter des données comportementales et utiliser des modèles d'apprentissage automatique pour identifier les anomalies. Il se compose de plusieurs éléments, notamment la collecte de données, l'ingénierie des fonctionnalités, la génération d'ensembles de données, la formation de modèles et la détection en temps réel.

## Fonctionnalités

- **Surveillance en temps réel** : Collecte des données à partir des processus système, de l'activité réseau et des événements du système de fichiers.
- **Analyse comportementale** : Utilise l'apprentissage automatique pour classer les comportements comme bénins ou malveillants.
- **Classification des menaces** : Identifie des types de menaces spécifiques tels que les rançongiciels (Ransomware), les enregistreurs de frappe (Keyloggers) et les balayages de ports (Port Scans).
- **Tableau de bord interactif** : Fournit une interface Web professionnelle pour la surveillance et l'analyse.
- **Outils CLI** : Interface en ligne de commande pour un contrôle et une automatisation efficaces.

## Configuration Requise

- Python 3.9+
- Docker (optionnel, pour le déploiement conteneurisé)

## Installation

1.  Cloner le dépôt :
    ```bash
    git clone https://github.com/simox6v/behavioral-detection.git
    cd behavioral-detection
    ```

2.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

Le système peut être géré à l'aide du script `run.py`.

### Lancement des Composants

-   **Tableau de bord** : Démarrer l'interface Web.
    ```bash
    python run.py dashboard
    ```

-   **Détecteur en temps réel** : Démarrer le moteur de détection.
    ```bash
    python run.py detector
    ```

-   **Générateur de données** : Générer des ensembles de données d'entraînement.
    ```bash
    python run.py generator
    ```

-   **Entraînement des modèles** : Entraîner les modèles d'apprentissage automatique.
    ```bash
    python run.py trainer
    ```

-   **Collecteur** : Démarrer le collecteur de données comportementales.
    ```bash
    python run.py collector
    ```

-   **Tout exécuter** : Démarrer tous les composants simultanément.
    ```bash
    python run.py all
    ```

-   **Lancer les tests** : Exécuter la suite de tests.
    ```bash
    python run.py tests
    ```

-   **Lancer** : Exécuter.
    ```bash
    .\.venv\Scripts\streamlit.exe run src\interface\streamlit_app.py
    ```
    

## Structure du Projet

-   `src/collector` : Modules pour la surveillance du comportement du système (Processus, Réseaux, Fichiers).
-   `src/detector` : Logique de détection en temps réel utilisant des modèles entraînés.
-   `src/features` : Pipelines d'ingénierie des fonctionnalités et de traitement des données.
-   `src/generator` : Scripts pour simuler des scénarios bénins et malveillants.
-   `src/models` : Entraînement, évaluation et gestion des modèles.
-   `src/interface` : Interfaces utilisateur (CLI et Tableau de bord Streamlit).
-   `config` : Fichiers de configuration.
-   `tests` : Tests unitaires et d'intégration.

## Licence

Yarbi nvalidiw
