import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. Configuration des modèles
model_badminton = YOLO('modele/best.pt')
model_pose = YOLO('yolov8n-pose.pt')

video_path = 'Video/Asia_Championship.mp4'
cap = cv2.VideoCapture(video_path)

# 2. SELECTION DES POINTS POUR L'HOMOGRAPHIE (4 Clics)
points_video = []

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_video.append([x, y])
        cv2.circle(frame_calib, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("CALIBRATION : Cliquez sur les 4 coins du 1/2 terrain du FOND", frame_calib)

ret, frame_calib = cap.read()
if ret:
    cv2.imshow("CALIBRATION : Cliquez sur les 4 coins du 1/2 terrain du FOND", frame_calib)
    cv2.setMouseCallback("CALIBRATION : Cliquez sur les 4 coins du 1/2 terrain du FOND", draw_circle)
    print("Cliquez dans l'ordre du demi-terrain du FOND : Fond Gauche, Fond Droit, Filet Droit, Filet Gauche. Puis appuyez sur une touche.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Dimensions réelles en cm d'un demi-terrain
largeur_terrain = 610
longueur_terrain = 670

points_terrain = np.array([
    [0, 0],                           # Fond Gauche
    [largeur_terrain, 0],             # Fond Droit
    [largeur_terrain, longueur_terrain], # Filet Droit
    [0, longueur_terrain]             # Filet Gauche
], dtype=np.float32)

# Calcul de la matrice d'homographie
H, _ = cv2.findHomography(np.array(points_video, dtype=np.float32), points_terrain)
H_inv = np.linalg.inv(H)

# Configuration des connexions du squelette (pour dessiner les lignes du corps)
CONNEXIONS_SQUELETTE = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Bras et épaules
    (5, 11), (6, 12), (11, 12),                # Tronc / Bassin
    (11, 13), (13, 15), (12, 14), (14, 16)     # Jambes et chevilles
]

# 3. INITIALISATION DES DEUX MATRICES DE HEATMAP
heatmap_fond = np.zeros((longueur_terrain, largeur_terrain), dtype=np.float32)
heatmap_devant = np.zeros((longueur_terrain, largeur_terrain), dtype=np.float32)

# 4. BOUCLE PRINCIPALE
res_badminton = model_badminton(video_path, stream=True, conf=0.25, imgsz=1080)
res_pose = model_pose(video_path, stream=True, conf=0.25, imgsz=1080)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for r_bad, r_pos in zip(res_badminton, res_pose):
    ret, frame = cap.read()
    if not ret:
        break

    boxes = r_bad.boxes.xyxy.cpu().numpy()
    classes = r_bad.boxes.cls.cpu().numpy()
    squelettes = r_pos.keypoints.xy.cpu().numpy()

    # Calque pour les points accumulés de la heatmap
    overlay = frame.copy()

    for box, cl in zip(boxes, classes):
        num_classe = int(cl)
        if num_classe == 0 or num_classe == 4:  # Ignorer filet et volant
            continue
        
        x1, y1, x2, y2 = map(int, box)
        
        # Définir la couleur du joueur (Vert pour J1, Bleu pour J2)
        couleur = (0, 255, 0) if num_classe == 1 else (255, 0, 0)

        # 🔥 A. DESSINER LA BOX DU JOUEUR
        cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
        cv2.putText(frame, f"Joueur {num_classe}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

        # Filtrage par le bassin pour trouver le bon squelette
        for squelette in squelettes:
            h_g = squelette[11]
            h_d = squelette[12]
            
            if h_g[0] > 0 or h_d[0] > 0:
                centre_x = int((h_g[0] + h_d[0]) / 2) if (h_g[0] > 0 and h_d[0] > 0) else int(max(h_g[0], h_d[0]))
                centre_y = int((h_g[1] + h_d[1]) / 2) if (h_g[1] > 0 and h_d[1] > 0) else int(max(h_g[1], h_d[1]))

                if x1 <= centre_x <= x2 and y1 <= centre_y <= y2:
                    
                    # 🔥 B. DESSINER LE SQUELETTE (Articulations + Os)
                    # 1. Dessiner les lignes (les os)
                    for start_idx, end_idx in CONNEXIONS_SQUELETTE:
                        pt1 = squelette[start_idx]
                        pt2 = squelette[end_idx]
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), couleur, 2)
                    
                    # 2. Dessiner les articulations (points jaunes pour le contraste)
                    for pt in squelette:
                        if pt[0] > 0 and pt[1] > 0:
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)

                    # Coordonnées des chevilles pour la heatmap
                    ch_g = squelette[15]
                    ch_d = squelette[16]
                    
                    if ch_g[0] > 0 and ch_d[0] > 0:
                        sol_x = (ch_g[0] + ch_d[0]) / 2
                        sol_y = (ch_g[1] + ch_d[1]) / 2

                        # Dessiner le point de contact au sol actuel (gros point rouge sous le joueur)
                        cv2.circle(frame, (int(sol_x), int(sol_y)), 6, (0, 0, 255), -1)

                        # Homographie
                        point_video = np.array([[[sol_x, sol_y]]], dtype=np.float32)
                        point_transforme = cv2.perspectiveTransform(point_video, H)
                        
                        real_x = int(point_transforme[0][0][0])
                        real_y = int(point_transforme[0][0][1])

                        # Enregistrement dans les matrices respectives
                        if num_classe == 2:
                            if 0 <= real_x < largeur_terrain and 0 <= real_y < longueur_terrain:
                                heatmap_fond[real_y, real_x] += 1
                        elif num_classe == 1:
                            fake_y = real_y - longueur_terrain
                            real_x_j1 = np.clip(largeur_terrain - real_x, 0, largeur_terrain - 1)
                            real_y_j1 = np.clip(fake_y, 0, longueur_terrain - 1)
                            heatmap_devant[int(real_y_j1), int(real_x_j1)] += 1
                    break

    # 🔥 C. REPROJETER ET DESSINER LES TRACES DE LA HEATMAP EN DIRECT
    # Joueur 2 (Points Bleus)
    pts_fond = np.argwhere(heatmap_fond > 0)
    if len(pts_fond) > 0:
        pts_real = np.array([[[p[1], p[0]]] for p in pts_fond], dtype=np.float32)
        pts_vid = cv2.perspectiveTransform(pts_real, H_inv)
        for pt in pts_vid:
            cv2.circle(overlay, (int(pt[0][0]), int(pt[0][1])), 2, (255, 0, 0), -1)

    # Joueur 1 (Points Verts)
    pts_devant = np.argwhere(heatmap_devant > 0)
    if len(pts_devant) > 0:
        pts_real_j1 = np.array([[[largeur_terrain - p[1], p[0] + longueur_terrain]] for p in pts_devant], dtype=np.float32)
        pts_vid_j1 = cv2.perspectiveTransform(pts_real_j1, H_inv)
        for pt in pts_vid_j1:
            cv2.circle(overlay, (int(pt[0][0]), int(pt[0][1])), 2, (0, 255, 0), -1)

    # Fusion transparente de l'overlay de la heatmap sur la vidéo principale
    frame_final = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Affichage du rendu final complet
    cv2.imshow("Badminton Vision - Full Analysis (Box + Squelette + Heatmap)", frame_final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 5. GÉNÉRATION DU GRAPHIQUE COMPARATIF MATPLOTLIB (Identique au précédent)
print("Génération du rapport tactique complet...")
heatmap_fond_smooth = cv2.GaussianBlur(heatmap_fond, (51, 51), 25)
heatmap_devant_smooth = cv2.GaussianBlur(heatmap_devant, (51, 51), 25)

if np.max(heatmap_fond_smooth) > 0: heatmap_fond_smooth /= np.max(heatmap_fond_smooth)
if np.max(heatmap_devant_smooth) > 0: heatmap_devant_smooth /= np.max(heatmap_devant_smooth)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
im1 = ax1.imshow(heatmap_fond_smooth, cmap='jet', extent=[0, 6.1, 0, 6.7], origin='lower')
ax1.set_title("Joueur 2 (Fond de court)")
ax1.set_xlabel("Largeur (m)")
ax1.set_ylabel("Longueur (m)")
fig.colorbar(im1, ax=ax1, label="Intensité")

im2 = ax2.imshow(heatmap_devant_smooth, cmap='jet', extent=[0, 6.1, 0, 6.7], origin='lower')
ax2.set_title("Joueur 1 (Premier plan)")
ax2.set_xlabel("Largeur (m)")
fig.colorbar(im2, ax=ax2, label="Intensité")

plt.suptitle("Analyse Tactique Comparative des deux joueurs (en mètres)")
plt.savefig("heatmap_comparative_badminton.png", dpi=300)
plt.show()