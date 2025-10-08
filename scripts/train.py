import os
import yaml
import torch
from models.biscript_model import create_biscript_model
from utils.data_loader import create_bilingual_dataset

def main():
    # Configuration
    config = {
        'base_model': 'yolov8n.pt',
        'num_classes': 1,
        'epochs': 50,
        'imgsz': 640,
        'batch_size': 8,
        'data_config': 'configs/bilingual_data.yaml',
        'project': 'runs/train',
        'name': 'biscript_yolo_v1'
    }
    
    print("🚀 Initialisation de l'entraînement BiScript-YOLO...")
    
    # Créer le modèle
    model = create_biscript_model(config)
    print("Modèle BiScript-YOLO créé avec succès")
    
    # Entraînement
    results = model.train(
        data=config['data_config'],
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch_size'],
        patience=10,
        project=config['project'],
        name=config['name'],
        save=True,
        exist_ok=True
    )
    
    print("Entraînement terminé!")
    return results

if __name__ == "__main__":
    main()
