import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =====================================================================
# CONFIGURATION & RÉGLAGES
# =====================================================================
CORRECTION_J1 = 0.12  
CORRECTION_J2 = 0.12  

model = YOLO('modele/best.pt')
video_path = 'Video/echange_court.mp4'

# --- ÉTAPE DE CALIBRATION PAR CLICS ---
coins_image = []

def selectionner_coins(event, x, y, flags, param):
    global coins_image
    if event == cv2.EVENT_LBUTTONDOWN:
        coins_image.append([x, y])
        # Dessiner un point pour voir où on a cliqué
        cv2.circle(img_calib, (x, y), 5, (0, 255, 255), -1)
        cv2.imshow("CALIBRATION : Cliquez sur les 4 coins du terrain", img_calib)

# Ouvrir la vidéo juste pour choper la première image
cap_calib = cv2.VideoCapture(video_path)
ret, img_calib = cap_calib.read()
cap_calib.release()

if not ret:
    print("❌ Impossible de lire la vidéo pour la calibration.")
    exit()

print("\n📍 FENÊTRE DE CALIBRATION OUVERTE")
print("Dans l'ordre, cliquez sur :")
print("1. Coin FOND GAUCHE | 2. Coin FOND DROIT | 3. Coin DEVANT DROIT | 4. Coin DEVANT GAUCHE")

cv2.namedWindow("CALIBRATION : Cliquez sur les 4 coins du terrain")
cv2.setMouseCallback("CALIBRATION : Cliquez sur les 4 coins du terrain", selectionner_coins)
cv2.imshow("CALIBRATION : Cliquez sur les 4 coins du terrain", img_calib)

# Attendre que l'utilisateur ait cliqué sur les 4 points
while len(coins_image) < 4:
    cv2.waitKey(1)
cv2.destroyAllWindows()

# Conversion des points cliqués en float32 pour OpenCV
pts_src = np.array(coins_image, dtype=np.float32)

# Vraies coordonnées correspondantes sur un terrain officiel (en mètres, origine 0,0 au filet)
# Un terrain fait 6.10m de large (-3.05 à 3.05) et 13.40m de long (-6.70 à 6.70)
pts_dst = np.array([
    [-3.05, -6.70],  # 1. Fond Gauche
    [3.05, -6.70],   # 2. Fond Droit
    [3.05, 6.70],    # 3. Devant Droit
    [-3.05, 6.70]    # 4. Devant Gauche
], dtype=np.float32)

# 🔥 LE CALCUL MAGIQUE : Calcul de la matrice d'homographie de perspective
matrice_homographie = cv2.getPerspectiveTransform(pts_src, pts_dst)

# --- REPRISE DE LA VIDÉO ---
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

calque_heatmap = np.zeros((height, width, 3), dtype=np.uint8)
j1_x_pix, j1_y_pix = [], []
j2_x_pix, j2_y_pix = [], []
j1_premier_point, j2_premier_point = True, True

print("🚀 Calibration réussie ! Lancement de la vidéo...")
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
            
            if cls_id == 1:
                couleur = (0, 0, 255)
                nom_joueur = "Joueur 1"
                pos_y = int(y2 + (hauteur_cadre * CORRECTION_J1))
                j1_x_pix.append(pos_x)
                j1_y_pix.append(pos_y)
            else:
                couleur = (255, 0, 0)
                nom_joueur = "Joueur 2"
                pos_y = int(y2 + (hauteur_cadre * CORRECTION_J2))
                j2_x_pix.append(pos_x)
                j2_y_pix.append(pos_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
            cv2.putText(frame, nom_joueur, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

            if 0 <= pos_x < width and 0 <= pos_y < height:
                if cls_id == 1 and j1_premier_point:
                    cv2.circle(calque_heatmap, (pos_x, pos_y), 12, (0, 0, 0), -1)
                    cv2.circle(calque_heatmap, (pos_x, pos_y), 9, (0, 255, 255), -1)
                    j1_premier_point = False
                elif cls_id == 2 and j2_premier_point:
                    cv2.circle(calque_heatmap, (pos_x, pos_y), 12, (0, 0, 0), -1)
                    cv2.circle(calque_heatmap, (pos_x, pos_y), 9, (255, 255, 0), -1)
                    j2_premier_point = False
                else:
                    cv2.circle(calque_heatmap, (pos_x, pos_y), 6, couleur, -1)

    video_finale = cv2.addWeighted(frame, 1.0, calque_heatmap, 0.6, 0)
    cv2.imshow("Badminton Vision", video_finale)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =====================================================================
# CONVERSION PAR MATRICE ET DESSIN DU PLOT
# =====================================================================
if len(j1_x_pix) > 0 or len(j2_x_pix) > 0:
    print("📐 Conversion mathématique des coordonnées via la matrice d'homographie...")
    
    # Fonction qui applique la matrice 3D sur un point pixel (X, Y)
    def appliquer_homographie(liste_x, liste_y):
        pts_m_x, pts_m_y = [], []
        for x, y in zip(liste_x, liste_y):
            # Transformation matricielle perspective
            point = np.array([x, y, 1.0], dtype=np.float32)
            point_transforme = np.dot(matrice_homographie, point)
            # Normalisation z (coordonnées homogènes)
            xm = point_transforme[0] / point_transforme[2]
            ym = point_transforme[1] / point_transforme[2]
            pts_m_x.append(xm)
            pts_m_y.append(ym)
        return pts_m_x, pts_m_y

    j1_x_m, j1_y_m = appliquer_homographie(j1_x_pix, j1_y_pix)
    j2_x_m, j2_y_m = appliquer_homographie(j2_x_pix, j2_y_pix)

    # --- DESSIN DU GRAPHIQUE ---
    fig, ax = plt.subplots(figsize=(7, 11))
    w_double, w_simple = 6.10 / 2, 5.18 / 2
    l_fond, l_service_court, l_service_long_double = 6.70, 1.98, 6.70 - 0.76
    c_court = '#0f62fe'
    
    ax.plot([-w_double, w_double, w_double, -w_double, -w_double], [-l_fond, -l_fond, l_fond, l_fond, -l_fond], color=c_court, linewidth=2, label="Terrain Officiel")
    ax.plot([-w_double, w_double], [0, 0], color='black', linewidth=3, label="Filet")
    ax.plot([-w_simple, -w_simple], [-l_fond, l_fond], color=c_court, linewidth=1, linestyle='--')
    ax.plot([w_simple, w_simple], [-l_fond, l_fond], color=c_court, linewidth=1, linestyle='--')
    ax.plot([-w_double, w_double], [-l_service_court, -l_service_court], color=c_court, linewidth=1.5)
    ax.plot([-w_double, w_double], [l_service_court, l_service_court], color=c_court, linewidth=1.5)
    ax.plot([-w_double, w_double], [-l_service_long_double, -l_service_long_double], color=c_court, linewidth=1)
    ax.plot([-w_double, w_double], [l_service_long_double, l_service_long_double], color=c_court, linewidth=1)
    ax.plot([0, 0], [-l_fond, -l_service_court], color=c_court, linewidth=1)
    ax.plot([0, 0], [l_service_court, l_fond], color=c_court, linewidth=1)

    if j1_x_m:
        ax.scatter(j1_x_m, j1_y_m, c='red', alpha=0.2, s=25, label='Déplacements J1')
        ax.scatter(j1_x_m[0], j1_y_m[0], c='yellow', edgecolors='black', s=150, linewidths=2, zorder=5, label='Départ J1')
    if j2_x_m:
        ax.scatter(j2_x_m, j2_y_m, c='blue', alpha=0.2, s=25, label='Déplacements J2 PERFECT')
        ax.scatter(j2_x_m[0], j2_y_m[0], c='cyan', edgecolors='black', s=150, linewidths=2, zorder=5, label='Départ J2')

    ax.set_title("📈 Suivi Mathématique Redressé (Homographie)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Largeur (Mètres)")
    ax.set_ylabel("Longueur (Mètres)")
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-7.5, 7.5)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('heatmap_metrique_officielle.png', dpi=300)
    print("🖼️ Graphique parfait sauvegardé !")
    plt.show()