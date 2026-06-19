from ultralytics import YOLO

# 1. On charge TON modèle entraîné sur Colab
model = YOLO('yolov8n-pose.pt')

# 2. On l'applique sur une vidéo locale
# conf=0.25 signifie qu'il affiche les détections s'il est sûr à au moins 25%
results = model.predict(source='../Video/mon_match.mp4', save=True, conf=0.25, show=True)

print("🔥 C'est fait ! Regarde le résultat dans le dossier runs/detect/predict/")