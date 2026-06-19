import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =====================================================================
# CONFIGURATION & RÉGLAGES
# =====================================================================
CORRECTION_J1 = 0.12  
CORRECTION_J2 = 0.12  

# 1. Charger ton modèle YOLO
model = YOLO('modele/best.pt')

# 2. Ouvrir la vidéo d'origine pour récupérer ses dimensions
video_path = 'Video/echange_court.mp4'
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 3. Créer le calque pour l'historique des points sur la vidéo
calque_heatmap = np.zeros((height, width, 3), dtype=np.uint8)

# Listes pour stocker l'historique pour le Plot final et le fichier TXT
j1_x, j1_y = [], []
j2_x, j2_y = [], []

# Variables pour mémoriser les premiers points de départ
j1_premier_point = True
j2_premier_point = True

print("🚀 Lancement de la vidéo... Regarde l'échange. (Appuie sur 'q' pour couper)")

# =====================================================================
# PARTIE 1 : LECTURE DE LA VIDÉO ET AFFICHAGE EN DIRECT
# =====================================================================
results = model.track(source=video_path, stream=True, conf=0.25)

for r in results:
    ret, frame = cap.read()
    if not ret:
        break
        
    if r.boxes is None:
        continue
        
    boxes = r.boxes.xyxy.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, clss):
        cls_id = int(cls)
        
        if cls_id in [1, 2]:
            x1, y1, x2, y2 = map(int, box)
            hauteur_cadre = y2 - y1
            pos_x = int((x1 + x2) / 2)
            
            # Choix des couleurs et correction de perspective
            if cls_id == 1:
                couleur = (0, 0, 255) # Rouge
                nom_joueur = "Joueur 1"
                pos_y = int(y2 + (hauteur_cadre * CORRECTION_J1))
                # Sauvegarde pour le plot futur
                j1_x.append(pos_x)
                j1_y.append(pos_y)
            else:
                couleur = (255, 0, 0) # Bleu
                nom_joueur = "Joueur 2"
                pos_y = int(y2 + (hauteur_cadre * CORRECTION_J2))
                # Sauvegarde pour le plot futur
                j2_x.append(pos_x)
                j2_y.append(pos_y)

            # A. Dessin du CADRE sur la vidéo
            cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
            cv2.putText(frame, nom_joueur, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

            # B. Dessin des POINTS sur la vidéo (avec détection du premier point)
            if 0 <= pos_x < width and 0 <= pos_y < height:
                if cls_id == 1:
                    if j1_premier_point:
                        cv2.circle(calque_heatmap, (pos_x, pos_y), 12, (0, 0, 0), -1)
                        cv2.circle(calque_heatmap, (pos_x, pos_y), 9, (0, 255, 255), -1) # Gros Jaune
                        j1_premier_point = False
                    else:
                        cv2.circle(calque_heatmap, (pos_x, pos_y), 6, couleur, -1)
                elif cls_id == 2:
                    if j2_premier_point:
                        cv2.circle(calque_heatmap, (pos_x, pos_y), 12, (0, 0, 0), -1)
                        cv2.circle(calque_heatmap, (pos_x, pos_y), 9, (255, 255, 0), -1) # Gros Cyan
                        j2_premier_point = False
                    else:
                        cv2.circle(calque_heatmap, (pos_x, pos_y), 6, couleur, -1)

    # Fusion et affichage de la vidéo
    video_finale = cv2.addWeighted(frame, 1.0, calque_heatmap, 0.6, 0)
    cv2.imshow("Badminton Vision - Analyse en cours", video_finale)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Fermeture propre de la vidéo
cap.release()
cv2.destroyAllWindows()
print("🎬 Vidéo terminée ou coupée. Génération des fichiers de sauvegarde...")

# =====================================================================
# PARTIE 2 : CRÉATION DU PLOT MATPLOTLIB & SAUVEGARDE TEXTE
# =====================================================================
# On vérifie qu'on a bien récupéré des données pour éviter de crash
if len(j1_x) > 0 or len(j2_x) > 0:
    fig, ax = plt.subplots(figsize=(8, 12))

    # Dessin du terrain simulé
    all_x = j1_x + j2_x
    all_y = j1_y + j2_y
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    ax.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], color='black', linewidth=2, label="Limites")
    milieu_y = (min_y + max_y) / 2
    ax.plot([min_x, max_x], [milieu_y, milieu_y], color='green', linestyle='--', linewidth=2, label="Filet")
    milieu_x = (min_x + max_x) / 2
    ax.plot([milieu_x, milieu_x], [min_y, max_y], color='gray', linestyle=':', linewidth=1)

    # Tracé des points sur le graphique
    if j1_x:
        ax.scatter(j1_x, j1_y, c='red', alpha=0.2, s=25, label='Déplacements J1')
        ax.scatter(j1_x[0], j1_y[0], c='yellow', edgecolors='black', s=150, linewidths=2, zorder=5, label='Départ J1')
    if j2_x:
        ax.scatter(j2_x, j2_y, c='blue', alpha=0.2, s=25, label='Déplacements J2')
        ax.scatter(j2_x[0], j2_y[0], c='cyan', edgecolors='black', s=150, linewidths=2, zorder=5, label='Départ J2')

    # Esthétique du plot
    ax.set_title("📊 Rapport de placement Tactique - Badminton", fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Sauvegarde de l'image du plot
    plt.tight_layout()
    plt.savefig('rapport_heatmap_badminton.png', dpi=300)
    print("🖼️ Fichier 'rapport_heatmap_badminton.png' sauvegardé !")

    # Sauvegarde du fichier texte
    with open('coordonnees_match.txt', 'w') as f:
        f.write("=== RAPPORT DE POSITIONNEMENT BADMINTON ===\n\n")
        f.write(f"J1 (Rouge) - Départ : X={j1_x[0] if j1_x else 'N/A'}, Y={j1_y[0] if j1_y else 'N/A'}\n")
        f.write(f"J2 (Bleu)  - Départ : X={j2_x[0] if j2_x else 'N/A'}, Y={j2_y[0] if j2_y else 'N/A'}\n\n")
        f.write("Frame | J1_X | J1_Y | J2_X | J2_Y\n")
        for i in range(max(len(j1_x), len(j2_x))):
            x1_v = j1_x[i] if i < len(j1_x) else "N/A"
            y1_v = j1_y[i] if i < len(j1_y) else "N/A"
            x2_v = j2_x[i] if i < len(j2_x) else "N/A"
            y2_v = j2_y[i] if i < len(j2_y) else "N/A"
            f.write(f"{i:04d}  | {x1_v}  | {y1_v}  | {x2_v}  | {y2_v}\n")
    print("📄 Fichier 'coordonnees_match.txt' sauvegardé !")
    
    # Affichage final du graphique à l'écran
    plt.show()
else:
    print("⚠️ Aucun joueur n'a été détecté, impossible de générer le rapport.")