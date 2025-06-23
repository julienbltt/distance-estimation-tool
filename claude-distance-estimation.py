#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programme d'Estimation de Distance Monoculaire pour Lunettes Intelligentes
Utilise YOLOv11 pour la détection de personnes et estimation de distance basée sur taille apparente
Optimisé pour usage en extérieur, plage 0.5-7 mètres
"""

import cv2
import numpy as np
import math
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Installation des dépendances nécessaires
# pip install ultralytics opencv-python numpy scipy

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("Erreur : Installez ultralytics avec : pip install ultralytics")
    exit(1)

@dataclass
class PersonDetection:
    """Structure pour stocker les informations d'une personne détectée"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    distance: Optional[float] = None
    filtered_distance: Optional[float] = None
    is_close: bool = False  # True si distance < 1m

class DistanceKalmanFilter:
    """Filtre de Kalman pour lisser les estimations de distance"""
    
    def __init__(self):
        # Initialisation du filtre de Kalman pour une dimension (distance)
        self.kalman = cv2.KalmanFilter(2, 1)  # 2 états (position, vitesse), 1 mesure
        
        # Matrice de mesure : on ne mesure que la position
        self.kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        
        # Matrice de transition : position += vitesse * dt
        self.kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        
        # Bruit de processus (incertitude du modèle)
        self.kalman.processNoiseCov = np.array([[0.01, 0], [0, 0.01]], np.float32)
        
        # Bruit de mesure (incertitude des mesures)
        self.kalman.measurementNoiseCov = np.array([[0.1]], np.float32)
        
        # Covariance d'erreur initiale
        self.kalman.errorCovPost = np.eye(2, dtype=np.float32)
        
        # État initial [position, vitesse]
        self.kalman.statePre = np.array([5.0, 0.0], np.float32)
        self.kalman.statePost = np.array([5.0, 0.0], np.float32)
        
        self.initialized = False
        
    def update(self, measurement: float) -> float:
        """Met à jour le filtre avec une nouvelle mesure"""
        if not self.initialized:
            self.kalman.statePost = np.array([measurement, 0.0], np.float32)
            self.initialized = True
            return measurement
            
        # Prédiction
        prediction = self.kalman.predict()
        
        # Correction avec la mesure
        corrected = self.kalman.correct(np.array([[measurement]], np.float32))
        
        return float(corrected[0])

class CameraCalibrator:
    """Classe pour calibrer automatiquement les paramètres de la caméra"""
    
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length_pixels = None
        self.calibration_samples = []
        
    def estimate_focal_length(self, fov_degrees: float = 75.0) -> float:
        """Estime la longueur focale basée sur le champ de vision"""
        diagonal_pixels = math.sqrt(self.image_width**2 + self.image_height**2)
        focal_length = diagonal_pixels / (2 * math.tan(math.radians(fov_degrees / 2)))
        return focal_length
    
    def calibrate_with_reference(self, bbox_height: int, real_distance: float, 
                                person_height: float = 1.7) -> float:
        """Calibre la caméra avec une personne de référence à distance connue"""
        focal_length = (bbox_height * real_distance) / person_height
        self.calibration_samples.append(focal_length)
        
        # Utilise la médiane des échantillons pour plus de robustesse
        if len(self.calibration_samples) >= 3:
            self.focal_length_pixels = np.median(self.calibration_samples)
        else:
            self.focal_length_pixels = focal_length
            
        return self.focal_length_pixels
    
    def get_focal_length(self) -> float:
        """Retourne la longueur focale calibrée ou estimée"""
        if self.focal_length_pixels is None:
            self.focal_length_pixels = self.estimate_focal_length()
        return self.focal_length_pixels

class MonocularDistanceEstimator:
    """Classe principale pour l'estimation de distance monoculaire"""
    
    def __init__(self, model_path: str = "yolo11n.pt", camera_index: int = 1):
        print("Initialisation du système d'estimation de distance...")
        
        # Chargement du modèle YOLO
        try:
            self.model = YOLO(model_path)
            print(f"Modèle YOLO chargé : {model_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            print("Téléchargement automatique du modèle YOLOv11n...")
            self.model = YOLO("yolov11n.pt")
        
        # Configuration de la caméra
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Résolution optimisée
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Vérification de l'ouverture de la caméra
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la caméra")
        
        # Obtention des dimensions réelles de la caméra
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Résolution caméra : {self.frame_width}x{self.frame_height}")
        
        # Initialisation du calibrateur
        self.calibrator = CameraCalibrator(self.frame_width, self.frame_height)
        
        # Paramètres anthropométriques
        self.AVERAGE_PERSON_HEIGHT = 1.70  # Hauteur moyenne d'une personne (mètres)
        self.HEIGHT_VARIATION = 0.15       # Variation de ±15cm
        
        # Seuils et paramètres
        self.DISTANCE_THRESHOLD = 1.0      # Seuil d'alerte (mètres)
        self.MIN_CONFIDENCE = 0.5          # Confiance minimale pour détection
        self.MAX_DISTANCE = 7.0            # Distance maximale considérée
        self.MIN_DISTANCE = 0.5            # Distance minimale considérée
        
        # Filtres de Kalman pour chaque personne détectée
        self.kalman_filters = {}
        self.person_trackers = {}
        self.next_person_id = 0
        
        # Historique pour stabilisation
        self.distance_history = deque(maxlen=10)
        
        # Variables pour les alertes
        self.alert_active = False
        self.alert_start_time = 0
        self.alert_duration = 2.0  # Durée minimale d'alerte (secondes)
        
        # Variables pour l'affichage
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("Système initialisé avec succès !")
        
    def calculate_distance(self, bbox_height: int, person_height: float = None) -> float:
        """Calcule la distance basée sur la hauteur de la boîte englobante"""
        if person_height is None:
            person_height = self.AVERAGE_PERSON_HEIGHT
            
        focal_length = self.calibrator.get_focal_length()
        
        # Formule de base : Distance = (Taille_Réelle × Longueur_Focale) / Taille_Pixels
        distance = (person_height * focal_length) / bbox_height
        
        # Limitation dans la plage valide
        distance = max(self.MIN_DISTANCE, min(distance, self.MAX_DISTANCE))
        
        return distance
    
    def track_person(self, bbox: Tuple[int, int, int, int], confidence: float) -> int:
        """Système simple de suivi de personnes basé sur la proximité des boîtes"""
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Recherche de la personne la plus proche dans les trackers existants
        min_distance = float('inf')
        best_match_id = None
        
        for person_id, (last_center, last_time) in self.person_trackers.items():
            if time.time() - last_time < 1.0:  # Timeout de 1 seconde
                dist = math.sqrt((center_x - last_center[0])**2 + (center_y - last_center[1])**2)
                if dist < min_distance and dist < 100:  # Seuil de distance pour matching
                    min_distance = dist
                    best_match_id = person_id
        
        if best_match_id is not None:
            # Mise à jour du tracker existant
            self.person_trackers[best_match_id] = ((center_x, center_y), time.time())
            return best_match_id
        else:
            # Création d'un nouveau tracker
            new_id = self.next_person_id
            self.next_person_id += 1
            self.person_trackers[new_id] = ((center_x, center_y), time.time())
            self.kalman_filters[new_id] = DistanceKalmanFilter()
            return new_id
    
    def cleanup_trackers(self):
        """Nettoie les trackers inactifs"""
        current_time = time.time()
        inactive_ids = []
        
        for person_id, (_, last_time) in self.person_trackers.items():
            if current_time - last_time > 2.0:  # Inactif depuis 2 secondes
                inactive_ids.append(person_id)
        
        for person_id in inactive_ids:
            del self.person_trackers[person_id]
            if person_id in self.kalman_filters:
                del self.kalman_filters[person_id]
    
    def detect_and_estimate(self, frame: np.ndarray) -> Tuple[np.ndarray, List[PersonDetection]]:
        """Détecte les personnes et estime leur distance"""
        # Détection YOLO
        results = self.model(frame, conf=self.MIN_CONFIDENCE, classes=[0])  # Classe 0 = personne
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extraction des coordonnées et confiance
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Calcul de la hauteur de la boîte
                    bbox_height = y2 - y1
                    
                    if bbox_height > 20:  # Filtrage des détections trop petites
                        # Calcul de la distance brute
                        raw_distance = self.calculate_distance(bbox_height)
                        
                        # Suivi de la personne
                        person_id = self.track_person((x1, y1, x2, y2), confidence)
                        
                        # Filtrage Kalman
                        if person_id in self.kalman_filters:
                            filtered_distance = self.kalman_filters[person_id].update(raw_distance)
                        else:
                            filtered_distance = raw_distance
                        
                        # Création de la détection
                        detection = PersonDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            distance=raw_distance,
                            filtered_distance=filtered_distance,
                            is_close=filtered_distance < self.DISTANCE_THRESHOLD
                        )
                        
                        detections.append(detection)
        
        # Nettoyage des trackers inactifs
        self.cleanup_trackers()
        
        return frame, detections
    
    def check_proximity_alert(self, detections: List[PersonDetection]) -> bool:
        """Vérifie s'il faut déclencher une alerte de proximité"""
        close_persons = [d for d in detections if d.is_close]
        
        if close_persons:
            if not self.alert_active:
                self.alert_active = True
                self.alert_start_time = time.time()
                print(f"🚨 ALERTE : {len(close_persons)} personne(s) à moins d'1 mètre !")
                
                # Son d'alerte (si speaker disponible)
                try:
                    # Génération d'un bip sonore simple
                    import os
                    if os.name == 'nt':  # Windows
                        import winsound
                        winsound.Beep(1000, 200)
                    else:  # Linux/Mac
                        os.system('echo -e "\a"')
                except:
                    pass
            
            return True
        else:
            # Désactivation de l'alerte après la durée minimale
            if self.alert_active and time.time() - self.alert_start_time > self.alert_duration:
                self.alert_active = False
                print("✅ Alerte désactivée - Zone sécurisée")
            
            return False
    
    def draw_annotations(self, frame: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """Dessine les annotations sur l'image"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            distance = detection.filtered_distance
            confidence = detection.confidence
            
            # Couleur selon la distance
            if detection.is_close:
                color = (0, 0, 255)  # Rouge pour alerte
                thickness = 3
            elif distance < 2.0:
                color = (0, 165, 255)  # Orange pour attention
                thickness = 2
            else:
                color = (0, 255, 0)  # Vert pour sécurisé
                thickness = 2
            
            # Dessin de la boîte englobante
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Texte avec distance et confiance
            label = f"{distance:.1f}m ({confidence:.0%})"
            
            # Fond du texte
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Texte blanc sur fond coloré
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Indicateur d'alerte
            if detection.is_close:
                cv2.putText(annotated_frame, "ALERTE!", (x1, y2 + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Informations système
        info_y = 30
        cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"Personnes detectees: {len(detections)}", (10, info_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Focal length pour debugging
        focal_length = self.calibrator.get_focal_length()
        cv2.putText(annotated_frame, f"Focal: {focal_length:.1f}px", (10, info_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Indicateur d'alerte globale
        if self.alert_active:
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 50), 
                         (0, 0, 255), -1)
            cv2.putText(annotated_frame, "🚨 PERSONNE TROP PROCHE 🚨", 
                       (annotated_frame.shape[1] // 2 - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return annotated_frame
    
    def update_fps(self):
        """Met à jour le compteur FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def calibrate_interactive(self):
        """Calibration interactive avec une personne de référence"""
        print("\n=== Calibration Interactive ===")
        print("Placez-vous à exactement 2 mètres de la caméra")
        print("Appuyez sur 'c' pour calibrer, 'q' pour ignorer")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Détection temporaire pour calibration
            results = self.model(frame, conf=0.3, classes=[0])
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox_height = y2 - y1
                        
                        # Affichage de la boîte de calibration
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, f"Hauteur: {bbox_height}px", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(frame, "Calibration - Placez-vous a 2m, appuyez 'c'", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Effectuer la calibration
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        box = boxes[0]  # Prendre la première détection
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox_height = y2 - y1
                        
                        focal_length = self.calibrator.calibrate_with_reference(
                            bbox_height, 2.0, self.AVERAGE_PERSON_HEIGHT
                        )
                        print(f"✅ Calibration effectuée ! Focal length: {focal_length:.1f}px")
                        cv2.destroyWindow('Calibration')
                        return
                print("❌ Aucune personne détectée pour calibration")
            
            elif key == ord('q'):
                print("Calibration ignorée - utilisation de l'estimation automatique")
                cv2.destroyWindow('Calibration')
                return
    
    def run(self, calibrate: bool = True):
        """Fonction principale d'exécution"""
        print("Démarrage du système d'estimation de distance...")
        
        # Calibration optionnelle
        if calibrate:
            self.calibrate_interactive()
        
        print("Système actif ! Appuyez sur 'q' pour quitter, 'r' pour recalibrer")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Erreur de lecture de la caméra")
                    break
                
                # Optimisation pour usage extérieur
                # Amélioration du contraste et de la luminosité
                frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
                
                # Détection et estimation
                start_time = time.time()
                processed_frame, detections = self.detect_and_estimate(frame)
                processing_time = time.time() - start_time
                
                # Vérification des alertes
                self.check_proximity_alert(detections)
                
                # Annotation de l'image
                annotated_frame = self.draw_annotations(processed_frame, detections)
                
                # Affichage du temps de traitement
                cv2.putText(annotated_frame, f"Traitement: {processing_time*1000:.1f}ms", 
                           (10, annotated_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Affichage
                cv2.imshow('Estimation Distance - Lunettes Intelligentes', annotated_frame)
                
                # Mise à jour FPS
                self.update_fps()
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.calibrate_interactive()
                elif key == ord('s'):
                    # Sauvegarde de l'image actuelle
                    timestamp = int(time.time())
                    filename = f"distance_estimation_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Image sauvegardée : {filename}")
        
        except KeyboardInterrupt:
            print("\nArrêt demandé par l'utilisateur")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Nettoyage des ressources"""
        print("Nettoyage des ressources...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Système arrêté proprement")

def main():
    """Fonction principale"""
    print("🤖 Système d'Estimation de Distance Monoculaire")
    print("Version optimisée pour lunettes intelligentes")
    print("=" * 50)
    
    try:
        # Initialisation du système
        estimator = MonocularDistanceEstimator()
        
        # Démarrage avec calibration
        estimator.run(calibrate=True)
        
    except Exception as e:
        print(f"Erreur système : {e}")
        print("Vérifiez que la caméra est connectée et accessible")
    
    finally:
        print("Fin du programme")

if __name__ == "__main__":
    main()
