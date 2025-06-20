# Installation: pip install ultralytics opencv-python
import cv2
from ultralytics import YOLO
import math

# Distance de la caméra à l'objet (visage) mesurée en centimètres
Known_distance = 60

# Largeur moyenne d'un visage dans le monde réel en centimètres
Known_width = 14.3

# Hauteur moyenne d'une personne en centimètres (pour la détection de personne complète)
Known_person_height = 170

# Couleurs
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)

# Police
fonts = cv2.FONT_HERSHEY_COMPLEX

# Charger le modèle YOLO
model = YOLO("yolo11n.pt")  # Modèle standard YOLO
face_model = None

# Essayer de charger un modèle spécialisé pour les visages si disponible
try:
    face_model = YOLO("yolo11n-face.pt")  # Modèle spécialisé visages (optionnel)
    print("Modèle YOLO visages chargé")
except:
    print("Utilisation du modèle YOLO standard pour les personnes")

def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    """Calcule la distance focale"""
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

def Distance_finder(Focal_Length, real_width, width_in_frame):
    """Calcule la distance basée sur la largeur de l'objet"""
    if width_in_frame == 0:
        return 0
    distance = (real_width * Focal_Length) / width_in_frame
    return distance

def get_person_data_yolo(image, use_face_detection=True):
    """
    Détecte toutes les personnes avec YOLO et retourne leurs dimensions individuelles
    """
    detection_info = []
    
    # Détection de personnes avec le modèle standard
    results = model(image, verbose=False, conf=0.4)
    
    person_id = 0
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                
                # Classe 0 = personne dans COCO dataset
                if class_id == 0:  
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > 0.4:
                        person_id += 1
                        
                        # Calculer les dimensions
                        person_width = x2 - x1
                        person_height = y2 - y1
                        
                        # Estimation précise de la largeur du visage pour cette personne
                        # Région de la tête (25% supérieur du corps)
                        head_region_height = person_height * 0.25
                        head_x1, head_y1 = x1, y1
                        head_x2, head_y2 = x2, y1 + head_region_height
                        
                        # Largeur du visage = environ 60% de la largeur de la tête
                        # Et largeur de la tête = environ 70% de la largeur des épaules
                        estimated_head_width = person_width * 0.7
                        estimated_face_width = estimated_head_width * 0.6
                        
                        # Couleur unique pour chaque personne
                        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
                        person_color = colors[(person_id - 1) % len(colors)]
                        
                        # Dessiner le rectangle autour de la personne
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), person_color, 2)
                        
                        # Dessiner la région de la tête
                        cv2.rectangle(image, (int(head_x1), int(head_y1)), 
                                    (int(head_x2), int(head_y2)), person_color, 1)
                        
                        # Étiquette avec ID de la personne
                        cv2.putText(image, f"P{person_id}", (int(x1), int(y1-10)), 
                                  fonts, 0.6, person_color, 2)
                        
                        detection_info.append({
                            'type': 'person',
                            'person_id': person_id,
                            'bbox': (x1, y1, x2, y2),
                            'head_region': (head_x1, head_y1, head_x2, head_y2),
                            'width': person_width,
                            'height': person_height,
                            'face_width': estimated_face_width,
                            'confidence': confidence,
                            'color': person_color
                        })
    
    # Essayer aussi la détection de visages si disponible
    if face_model and use_face_detection:
        face_results = face_model(image, verbose=False, conf=0.3)
        
        face_id = 0
        for result in face_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > 0.3:
                        face_id += 1
                        
                        # Calculer la largeur du visage
                        face_width = x2 - x1
                        
                        # Couleur spéciale pour les visages détectés directement
                        face_color = (0, 255, 255)  # Cyan
                        
                        # Dessiner le rectangle autour du visage
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), face_color, 2)
                        
                        # Étiquette
                        cv2.putText(image, f"F{face_id}", (int(x1), int(y1-10)), 
                                  fonts, 0.5, face_color, 2)
                        
                        detection_info.append({
                            'type': 'face',
                            'face_id': face_id,
                            'bbox': (x1, y1, x2, y2),
                            'width': face_width,
                            'face_width': face_width,
                            'confidence': confidence,
                            'color': face_color
                        })
    
    return detection_info

def draw_distance_info(image, distance, detection_type, x, y, confidence, person_id, color):
    """Dessine les informations de distance sur l'image avec ID de personne"""
    
    # Calculer la position du texte pour éviter les chevauchements
    text_y_offset = 0
    if detection_type == "Visage":
        text_y_offset = -80
    else:
        text_y_offset = -60
    
    # Ligne de fond pour le texte avec la couleur de la personne
    text_bg_y = int(y + text_y_offset)
    cv2.line(image, (int(x), text_bg_y), (int(x+280), text_bg_y), color, 40)
    cv2.line(image, (int(x), text_bg_y), (int(x+280), text_bg_y), BLACK, 36)
    
    # Texte de distance avec ID
    if detection_type == "Visage":
        label = f"F{person_id}"
    else:
        label = f"P{person_id}"
    
    if distance < 100:
        text = f"{label} - Distance: {distance:.2f} CM"
    else:
        text = f"{label} - Distance: {distance/100:.3f} M"
    
    cv2.putText(image, text, (int(x), text_bg_y + 5), fonts, 0.5, WHITE, 2)
    
    # Confiance
    conf_text = f"Conf: {confidence:.2f}"
    cv2.putText(image, conf_text, (int(x), text_bg_y + 25), fonts, 0.4, WHITE, 1)

def main():
    # Lire l'image de référence
    ref_image_path = "captured_images/capture_20250620_155553_000.jpg"
    ref_image = cv2.imread(ref_image_path)
    
    if ref_image is None:
        print(f"Erreur: Impossible de charger l'image de référence {ref_image_path}")
        print("Assurez-vous que le fichier existe ou changez le chemin")
        return
    
    print("Analyse de l'image de référence...")
    
    # Trouver la largeur du visage (en pixels) dans l'image de référence
    ref_detections = get_person_data_yolo(ref_image)
    
    if not ref_detections:
        print("Aucune personne détectée dans l'image de référence!")
        print("Assurez-vous qu'il y a une personne visible dans l'image")
        return
    
    # Utiliser la première personne détectée comme référence
    ref_face_width = ref_detections[0]['face_width']
    print(f"Largeur du visage de référence: {ref_face_width:.1f} pixels")
    print(f"Personnes détectées dans la référence: {len(ref_detections)}")
    
    # Calculer la distance focale
    Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_face_width)
    print(f"Distance focale calculée: {Focal_length_found:.6f}")
    
    # Afficher l'image de référence
    cv2.imshow("Image de reference", ref_image)
    
    # Initialiser la caméra
    cap = cv2.VideoCapture(1)  # Changez en 1 si nécessaire
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return
    
    print("Caméra initialisée. Appuyez sur 'q' pour quitter")
    print("Appuyez sur 'c' pour recalibrer avec l'image actuelle")
    
    while True:
        # Lire la frame depuis la caméra
        ret, frame = cap.read()
        if not ret:
            break
        
        # Détecter toutes les personnes dans la frame actuelle
        detections = get_person_data_yolo(frame)
        
        # Calculer et afficher la distance pour chaque personne détectée
        for detection in detections:
            confidence = detection['confidence']
            person_color = detection['color']
            
            if detection['type'] == 'face':
                width_for_distance = detection['face_width']
                distance = Distance_finder(Focal_length_found, Known_width, width_for_distance)
                draw_distance_info(frame, distance, "Visage", 
                                 detection['bbox'][0], detection['bbox'][1], 
                                 confidence, detection['face_id'], person_color)
            
            elif detection['type'] == 'person':
                width_for_distance = detection['face_width']
                distance = Distance_finder(Focal_length_found, Known_width, width_for_distance)
                draw_distance_info(frame, distance, "Personne", 
                                 detection['bbox'][0], detection['bbox'][1], 
                                 confidence, detection['person_id'], person_color)
        
        # Afficher les informations générales
        cv2.putText(frame, f"Focale: {Focal_length_found:.3f}", (10, 30), 
                   fonts, 0.5, WHITE, 1)
        cv2.putText(frame, f"Personnes detectees: {len([d for d in detections if d['type'] == 'person'])}", (10, 50), 
                   fonts, 0.5, WHITE, 1)
        cv2.putText(frame, f"Visages detectes: {len([d for d in detections if d['type'] == 'face'])}", (10, 70), 
                   fonts, 0.5, WHITE, 1)
        cv2.putText(frame, "C=Calibrer | Q=Quitter | S=Stats", (10, frame.shape[0]-30), 
                   fonts, 0.5, WHITE, 1)
        
        # Afficher la frame
        cv2.imshow("YOLO Distance Estimation", frame)
        
        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            # Recalibrer avec l'image actuelle
            print("\n=== RECALIBRATION ===")
            if detections:
                print("Personnes détectées:")
                for i, det in enumerate(detections):
                    if det['type'] == 'person':
                        print(f"  P{det['person_id']}: largeur visage = {det['face_width']:.2f}px")
                    else:
                        print(f"  F{det['face_id']}: largeur visage = {det['face_width']:.2f}px")
                
                person_choice = input("Entrez l'ID de la personne pour calibration (P1, P2, F1, etc.): ")
                distance_input = input("Entrez la distance actuelle de cette personne en CM: ")
                
                try:
                    new_distance = float(distance_input)
                    
                    # Trouver la personne sélectionnée
                    selected_detection = None
                    for det in detections:
                        if det['type'] == 'person' and f"P{det['person_id']}" == person_choice.upper():
                            selected_detection = det
                            break
                        elif det['type'] == 'face' and f"F{det['face_id']}" == person_choice.upper():
                            selected_detection = det
                            break
                    
                    if selected_detection:
                        Focal_length_found = Focal_Length_Finder(new_distance, Known_width, 
                                                                selected_detection['face_width'])
                        print(f"Nouvelle distance focale: {Focal_length_found:.6f}")
                        print("Calibration terminée!")
                    else:
                        print("Personne non trouvée, calibration annulée")
                        
                except ValueError:
                    print("Distance invalide, calibration annulée")
            else:
                print("Aucune personne détectée pour la calibration")
            print("===================\n")
                
        elif key == ord("s"):
            # Afficher les statistiques détaillées pour chaque personne
            print(f"\n=== STATISTIQUES DÉTAILLÉES ===")
            print(f"Distance focale: {Focal_length_found:.6f}")
            print(f"Largeur visage référence: {ref_detections[0]['face_width']:.3f} pixels")
            print(f"Personnes détectées: {len([d for d in detections if d['type'] == 'person'])}")
            print(f"Visages détectés: {len([d for d in detections if d['type'] == 'face'])}")
            
            for det in detections:
                distance = Distance_finder(Focal_length_found, Known_width, det['face_width'])
                if det['type'] == 'person':
                    print(f"  P{det['person_id']}: {distance:.3f}cm (conf: {det['confidence']:.3f}, largeur: {det['face_width']:.2f}px)")
                else:
                    print(f"  F{det['face_id']}: {distance:.3f}cm (conf: {det['confidence']:.3f}, largeur: {det['face_width']:.2f}px)")
            print("===============================\n")
    
    # Nettoyer
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()