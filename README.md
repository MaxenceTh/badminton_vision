# 🏸 Badminton Vision — Tracking Tactique par IA

Ce projet utilise la vision par ordinateur (YOLO) pour automatiser l'analyse tactique d'un match de badminton. Il détecte les joueurs et redresse la perspective de la caméra pour générer des heatmaps à l'échelle métrique réelle (en mètres) sur un terrain officiel.

---

## 🚀 Fonctionnalités
<!-- * **Détection en direct :** Tracking du filet, du volant et distinction automatique entre le `Joueur 1` (premier plan) et le `Joueur 2` (fond de court).
* **Calibration par Homographie :** Système graphique par clics pour corriger l'écrasement 3D de la caméra.
* **Rapport Tactique :** Export d'un graphique Matplotlib (`.png`) aux dimensions officielles de la BWF (Fédération Internationale de Badminton).
* **Données Brutes :** Sauvegarde des coordonnées $(X, Y)$ réelles dans un fichier texte pour de futures analyses. -->

---

## 📦 Installation & Environnement

### 1. Cloner le projet & Configurer Git
```bash
git clone [https://github.com/MaxenceTh/badminton_vision.git](https://github.com/MaxenceTh/badminton_vision.git)
cd "Badminton vision"
```

### 2. Créer et activer environnement virtuel
```bash
# Création
python -m venv env_badminton
# Activation (Windows PowerShell)
.\env_badminton\Scripts\activate
# Activation (Linux / macOS)
source env_badminton/bin/activate
```

### 3. Installer les dépendances
```bash
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy matplotlib
```

## Source et Dataset
- https://universe.roboflow.com/mathieu-cartron/yolov5-swgec
- https://www.youtube.com/watch?v=r0RspiLG260&t=1107s
- https://universe.roboflow.com/highlightsportbt/badmintoncourtdetectionoffical-b3hl9
