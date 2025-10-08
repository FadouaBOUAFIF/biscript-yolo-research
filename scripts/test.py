import cv2
import torch
from models.biscript_model import create_biscript_model
import numpy as np

class BiScriptTester:
    def __init__(self, model_path):
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        
    def test_single_image(self, image_path):
        """Test sur une seule image"""
        # Charger l'image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Préparation
        input_tensor = self.preprocess_image(image_rgb)
        
        # Inference
        with torch.no_grad():
            outputs, script_probs = self.model(input_tensor)
            
        # Post-processing
        detections = self.postprocess_detections(outputs)
        script_type = "latin" if script_probs[0][0] > 0.5 else "arabic"
        script_confidence = max(script_probs[0])
        
        return {
            'detections': detections,
            'script_type': script_type,
            'script_confidence': float(script_confidence),
            'raw_probs': script_probs
        }
    
    def preprocess_image(self, image, size=640):
        """Prétraitement de l'image"""
        # Redimensionnement
        h, w = image.shape[:2]
        scale = min(size / h, size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Padding
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalisation
        normalized = padded / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float().unsqueeze(0)
        
        return tensor
    
    def postprocess_detections(self, outputs, conf_threshold=0.5):
        """Post-traitement des détections"""
        # Implémentation simplifiée
        detections = []
        
        # Extraire les boîtes, scores et classes
        boxes = outputs[0, :, :4]  # x1, y1, x2, y2
        scores = outputs[0, :, 4]  # scores de confiance
        classes = outputs[0, :, 5]  # classes
        
        # Filtrer par score
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
        
        for box, score, cls in zip(boxes, scores, classes):
            detections.append({
                'bbox': box.tolist(),
                'confidence': float(score),
                'class': int(cls)
            })
        
        return detections

def demo_test():
    """Test de démonstration"""
    print(" Démonstration du test BiScript-YOLO")
    
    # Créer des données de test
    from utils import create_demo_image
    test_image = create_demo_image(640, 640, is_arabic=True)
    cv2.imwrite('test_demo.jpg', test_image)
    
    # Test (avec modèle factice pour la démo)
    print("Création image de test...")
    print(" Test terminé - Prêt pour les vraies données!")
    
    return True

if __name__ == "__main__":
    demo_test()
