## 1. SUPPORT TECHNIQUE

### 1.1 Présentation
- **Nom du projet** : RAG Chatbot interne
- **Objectif** : Fournir un service de chatbot RAG (Retrieval-Augmented Generation) déployé en interne, accessible via FastAPI et Streamlit.
- **Fonctionnalités principales** :
  - Endpoints FastAPI pour `/`, `/retrieve`, `/chat`, `/upload`
  - Interface Streamlit modulable en 5 pages (Home, Document Mining, Add Documents, Chat, Dashboard)
  - Mécanisme de file d'attente asynchrone et historique de chat stocké dans Redis
  - Collecte de feedback utilisateurs et métriques d’usage
- **Nom, port et version du serveur** : 
  - Nom : `FRSDCPS1V1076.corp.idemia.com`
  - Port : `49100`
  - Version : `Windows Server 2019`
- **Addresse** : http://10.234.1.32
- **Ports** : 
  - UI (Streamlit) : 8501
  - Fast API : 8000 (ajouté `/docs` pour l'interface) -> `8000/docs`

**Donc pour utiliser l'application , tapez sur votre navigateur [http://10.234.1.32:8501](http://10.234.1.32:8501).**

### 1.2 Prérequis
- Windows Server 2019
- Python 3.11 et dépendances listées dans l’environnement Conda (`rag.yaml`)
- Redis installé comme service Windows
- Win32 NSSM (Non-Sucking Service Manager) installé dans `C:\Users\t1appbse\Documents\nssm-2.24\win32`

### 1.3 Installation et lancement
1. Cloner le dépôt dans `C:\CPS\version 7`
2. Créer et activer l’environnement Conda :
   ```powershell
   conda env create -f rag.yaml
   conda activate rag
   # cet étape n'est pas forcément nécessaire vu qu'il est inclus dans le service , donc cela se fait automatiquement au démarrage du serveur.
   ```
3. Installer et démarrer Redis :
   ```powershell
   cd C:\Users\t1appbse\Documents\Redis-x64-3.0.504
   # pas nécessaire vu qu'il est déjà installé.
   .\redis-server.exe --service-install redis.windows-service.conf --loglevel verbose

   # Lancé le service au cas où il s'arrête pour une raison inconnue
   .\redis-server.exe --service-start 
   ```
4. Installer et configurer NSSM pour FastAPI et Streamlit :
   ```powershell
   cd C:\Users\t1appbse\Documents\nssm-2.24\win32
   .\nssm.exe install FastApi 
   .\nssm.exe install StreamlitApp
   # Cet étape n'est nécessaire, les services sont déjà créées et tourne dans le serveur en mode 'automatic'
   ```
5. Démarrer les services :
   ```powershell
   # FastAPI et Streamlit
   cd C:\Users\t1appbse\Documents\nssm-2.24\win32
   .\nssm.exe start FastApi
   .\nssm.exe start StreamlitApp
   # Redis
   cd C:\Users\t1appbse\Documents\Redis-x64-3.0.504
   .\redis-server.exe --service-start

   #  Veillez noter que Redis et Nssm ne se trouve pas dans le répertoire partagé CPS mais plutôt dans Documnents.
   ```

---

## 2. Architecture

### 2.1 Structure du projet
```
C:/CPS/version 7
├─ fast_api_app.py       # Application FastAPI (endpoints, worker async)
├─ generator.py          # Génération de réponses Llama via LlamaCpp
├─ retriever.py          # Chargement et requête du vectorstore FAISS
├─ redis_db.py           # Gestion de l'historique et feedback via Redis
├─ preprocess.py         # Prétraitement et indexation des documents
├─ paths.py              # Variables de chemins pour l’application
├─ rag.yaml              # Environnement Conda
├─ streamlit_app.py      # Point d’entrée de l’interface Streamlit
├─ streamlit_pages/      # Pages modulaires de l’UI Streamlit
│  ├─ home.py            # Page d’accueil et navigation
│  ├─ document_mining.py # Recherche et visualisation PDF
│  ├─ documents.py       # Upload et ingestion de PDF
│  ├─ chatbot.py         # Interface de chat avec feedback
│  └─ dashboard.py       # Visualisation des métriques et feedbacks
├─ uploads/              # Dossier pour fichiers téléchargés
└─ preprocessed_data/    # Index FAISS, chunks pickle, etc.
```

### 2.2 Flux des requêtes
1. **Client** → **Streamlit** ou **API** ➔ FastAPI
2. Requête **Retrieve** / **Chat**
3. File d’attente async + **lock** pour modèle
4. **Llama** via LlamaCpp ↔ **Redis** (historique)
5. Réponse retournée au client

---

## 3. Référence API (FastAPI)

| Endpoint      | Méthode | Description                                      | Paramètres                          |
|---------------|---------|--------------------------------------------------|-------------------------------------|
| `/`           | GET     | Vérifie que l’API est en ligne                   | —                                   |
| `/retrieve`   | POST    | Retourne les documents pertinents                | `query: str`                        |
| `/chat`       | POST    | Traite un message de chat                        | `user_input: str`, `session_id: str`|
| `/upload`     | POST    | Sauvegarde un fichier                            | `file: UploadFile`                  |

---

## 4. Déploiement & exploitation

### 4.1 Démarrage, arrêt et redémarrage des services

- **FastAPI & Streamlit** (via NSSM)
  ```powershell
  cd C:\Users\t1appbse\Documents\nssm-2.24\win32
  .\nssm.exe start FastApi
  .\nssm.exe stop FastApi
  .\nssm.exe restart FastApi

  .\nssm.exe start StreamlitApp
  .\nssm.exe stop StreamlitApp
  .\nssm.exe restart StreamlitApp
  ```
- **Redis**
  ```powershell
  cd C:\Users\t1appbse\Documents\Redis-x64-3.0.504
  .\redis-server.exe --service-start
  .\redis-server.exe --service-stop
  # Pas de restart, faire stop puis start
  ```

### 4.2 Logs

- Emplacement des logs :
  ```text
  C:\CPS\logs
  ```
- Contenu : fichiers `.log` pour chaque service (erreurs et sorties).

---

## 5. Support technique

### 5.1 Surveillance et alerting
- **Métriques clés** : latence, erreurs 5xx, CPU/mémoire
- **Outils** : Windows Performance Monitor, Dashboard de l'application (visible pour les administrateurs)

### 5.2 Maintenance & mises à jour
- **Code** : modifications dans `C:\CPS\version 7` rechargées automatiquement
- **Rollback** : stop service, restart via NSSM

### 5.3 Procédure d’incident
1. Identifier la partie de l'application qui est impactée
2. Identifier le Service concerné
   1. `FastApi` (le backend)
   2. `StreamlitApp` (l'interface web)
   3. `Redis` (la base de données)
3. Consulter les logs dans `C:\CPS\logs` du service concerné , identifier le problème et le corriger (si c'est en rapport avec le code, allez sur  `C:\CPS\version 7` et regardez les sections `2.1 Structure du projet` et `6. Documentation de l’interface Streamlit` de ce même support pour comprendre le rôle de chaque fichier du code)
4. Redémarrer le service (NSSM ou Redis)

---

## 6. Documentation de l’interface Streamlit

### 6.1 home.py
- Affiche le logo et le titre
- Présente les fonctionnalités (Document Mining, Add Documents, Chat, Dashboard)
- Liens de navigation vers chaque page

### 6.2 document_mining.py
- Recherche sémantique via `/retrieve`
- Visualisation des résultats PDF
- Intégration du visionneur PDF (`streamlit_pdf_viewer`)
- Explication et vidéo d’aide (chemin : `paths.exp_video`)

### 6.3 documents.py
- Upload de fichiers PDF
- Stockage dans `uploads/`
- Confirmation de succès via `st.success`

### 6.4 chatbot.py
- Envoi de requêtes `/chat` à FastAPI
- Affichage streaming mot-à-mot et durée de génération
- Stockage des interactions dans `st.session_state.history`
- Collecte de feedback (pouces) et stockage Redis
- Blocage/déblocage bouton pour éviter doublons

### 6.5 dashboard.py
- Récupération des métriques utilisateurs et feedbacks pour les 7 derniers jours
- Graphiques : bar_chart (utilisateurs), bar_chart (feedbacks pos/neg), area_chart (taux feedback)
- KPI de taux de feedback actuel via `st.metric`

---

## 7. Tests & CI
- **unit_test.py** : tests pytest pour fonctions clés (prétraitement, récupération, historique Redis)
- Exécution :
  ```bash
  pytest unit_test.py
  ```

---

## 8. Contributing & Changelog
- **CONTRIBUTING.md** : guide de contribution (format de commit, revue, branches)
- **CHANGELOG.md** : historique des versions et changements majeurs
