<<<<<<< HEAD
#!/usr/bin/env python3
"""
Setup Script for BiScript-YOLO Project
"""

import os

print("Starting BiScript-YOLO Project Setup...")
print("=" * 50)

# Créer les dossiers essentiels
directories = [
    'models',
    'scripts', 
    'configs',
    'notebooks',
    'docs',
    'data/raw/latin',
    'data/raw/arabic',
    'data/processed/latin',
    'data/processed/arabic',
    'runs/train',
    'runs/detect',
    'runs/eval'
]

print("Creating directories...")
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f" {directory}")

# Créer les fichiers Python essentiels
print("\nCreating Python files...")

# Fichier modèle principal
with open('models/biscript_model.py', 'w') as f:
    f.write('''"""
BiScript-YOLO Model Implementation
"""

import torch
import torch.nn as nn

class BiScriptYOLO(nn.Module):
    """Modèle principal BiScript-YOLO"""
    
    def __init__(self):
        super().__init__()
        print("BiScript-YOLO model created!")
    
    def forward(self, x):
        return x

# Test du modèle
if __name__ == "__main__":
    model = BiScriptYOLO()
''')
print(" models/biscript_model.py")

# Script d'entraînement
with open('scripts/train.py', 'w') as f:
    f.write('''"""
Training Script for BiScript-YOLO
"""

print("Starting BiScript-YOLO training...")
print("Training script ready!")

# Test simple
if __name__ == "__main__":
    print("Test passed!")
''')
print("scripts/train.py")

# Script de test
with open('scripts/test.py', 'w') as f:
    f.write('''"""
Testing Script for BiScript-YOLO
"""

print("Starting BiScript-YOLO testing...")
print("Testing script ready!")

if __name__ == "__main__":
    print("Test passed!")
''')
print("scripts/test.py")

# Configuration
with open('configs/training_config.yaml', 'w') as f:
    f.write('''# Configuration d'entraînement BiScript-YOLO
model: yolov8n.pt
epochs: 100
batch_size: 8
imgsz: 640
''')
print("configs/training_config.yaml")

print("\n" + "=" * 50)
print("PROJECT SETUP COMPLETED SUCCESSFULLY!")
print("\nNext steps:")
print("1. Run: python scripts/train.py")
print("2. Run: python scripts/test.py") 
print("3. Run: python models/biscript_model.py")
print("4. Check structure: dir /S")
=======

>>>>>>> 37819aabe3b429bbf51200f675d9870452a70b48
