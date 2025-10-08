import os
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

def convert_to_yolo_format(input_dir, output_dir):
    """Convertit les annotations au format YOLO"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Pour la démo, création d'annotations factices
    create_demo_dataset(output_dir)

def create_demo_dataset(output_dir, num_samples=100):
    """Crée un dataset de démonstration pour tester"""
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    for i in range(num_samples):
        # Créer image factice (dans la réalité, utiliser vos vraies images)
        img = create_demo_image(640, 640, is_arabic=(i % 2 == 0))
        img_path = os.path.join(images_dir, f'image_{i:04d}.jpg')
        cv2.imwrite(img_path, img)
        
        # Créer annotations factices
        label_path = os.path.join(labels_dir, f'image_{i:04d}.txt')
        with open(label_path, 'w') as f:
            # Format YOLO: class x_center y_center width height
            f.write("0 0.5 0.5 0.3 0.2\n")  # Zone manuscrite

def create_demo_image(width, height, is_arabic=False):
    """Crée une image de démonstration avec du texte"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Ajouter du texte "manuscrit" factice
    if is_arabic:
        # Simuler texte arabe (courbes)
        for j in range(5):
            y = 100 + j * 80
            cv2.ellipse(img, (320, y), (200, 10), 0, 0, 360, (0, 0, 0), 2)
    else:
        # Simuler texte latin (lignes droites)
        for j in range(5):
            y = 100 + j * 80
            cv2.line(img, (150, y), (490, y), (0, 0, 0), 2)
    
    return img

def create_data_yaml(output_path):
    """Crée le fichier de configuration YOLO"""
    data = {
        'path': os.path.abspath(output_path),
        'train': 'images',
        'val': 'images',  # Pour la démo, même données
        'nc': 1,
        'names': ['handwritten_zone']
    }
    
    with open(output_path + '/dataset.yaml', 'w') as f:
        yaml.dump(data, f)
