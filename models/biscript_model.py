import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Detect
from ultralytics.utils.loss import v8DetectionLoss
import math

class ScriptRouter(nn.Module):
    """Module léger pour la détection de script"""
    def __init__(self, input_channels=256, num_scripts=2):
        super().__init__()
        
        # Réduction dimensionnelle progressive
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        
        # Classificateur de script
        self.script_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_scripts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        features = self.conv_layers(x)
        script_probs = self.script_classifier(features)
        return script_probs

class BiScriptDetectionHead(Detect):
    """Tête de détection bi-script modifiée"""
    def __init__(self, nc=80, anchors=(), ch=(), router_channels=256):
        super().__init__(nc, anchors, ch)
        
        # Script Router
        self.script_router = ScriptRouter(input_channels=router_channels)
        
        # Duplication des têtes de détection pour chaque script
        self.nl = len(anchors)  # Number of detection layers
        self.na = len(anchors[0]) // 2  # Number of anchors
        self.no = nc + 5  # Number of outputs per anchor
        
        # Têtes séparées pour latin et arabe
        self.latin_cv2 = nn.ModuleList()
        self.arabic_cv2 = nn.ModuleList()
        self.latin_cv3 = nn.ModuleList()
        self.arabic_cv3 = nn.ModuleList()
        
        for i in range(self.nl):
            # Tête latine
            self.latin_cv2.append(nn.Conv2d(ch[i], self.na * self.no, 1))
            self.latin_cv3.append(nn.Conv2d(ch[i], self.na * self.no, 1))
            
            # Tête arabe
            self.arabic_cv2.append(nn.Conv2d(ch[i], self.na * self.no, 1))
            self.arabic_cv3.append(nn.Conv2d(ch[i], self.na * self.no, 1))
        
        # Initialisation des poids
        self._initialize_bi_script_weights()
    
    def _initialize_bi_script_weights(self):
        """Initialisation des poids pour les deux têtes"""
        for i in range(self.nl):
            # Copier les poids initiaux de la tête originale
            cv2_weight = self.cv2[i].weight.data.clone()
            cv3_weight = self.cv3[i].weight.data.clone()
            
            # Initialiser les têtes latines
            self.latin_cv2[i].weight.data.copy_(cv2_weight)
            self.latin_cv3[i].weight.data.copy_(cv3_weight)
            
            # Initialiser les têtes arabes avec une petite variation
            self.arabic_cv2[i].weight.data.copy_(cv2_weight + 0.01 * torch.randn_like(cv2_weight))
            self.arabic_cv3[i].weight.data.copy_(cv3_weight + 0.01 * torch.randn_like(cv3_weight))
    
    def forward(self, x):
        """Forward pass avec routing de script"""
        script_probs = self.script_router(x[0])  # Utilise la première feature map
        
        # Détections pour chaque script
        latin_outputs = []
        arabic_outputs = []
        
        for i in range(self.nl):
            # Latin
            latin_reg_outputs = self.latin_cv2[i](x[i])
            latin_cls_outputs = self.latin_cv3[i](x[i])
            latin_outputs.append(torch.cat([latin_reg_outputs, latin_cls_outputs], 1))
            
            # Arabe
            arabic_reg_outputs = self.arabic_cv2[i](x[i])
            arabic_cls_outputs = self.arabic_cv3[i](x[i])
            arabic_outputs.append(torch.cat([arabic_reg_outputs, arabic_cls_outputs], 1))
        
        # Fusion basée sur les probabilités de script
        batch_size = script_probs.shape[0]
        latin_weight = script_probs[:, 0].view(batch_size, 1, 1, 1, 1)  # Poids pour latin
        arabic_weight = script_probs[:, 1].view(batch_size, 1, 1, 1, 1)  # Poids pour arabe
        
        # Fusion pondérée
        fused_outputs = []
        for lat_out, arab_out in zip(latin_outputs, arabic_outputs):
            fused = latin_weight * lat_out + arabic_weight * arab_out
            fused_outputs.append(fused)
        
        return torch.cat([x.flatten(2) for x in fused_outputs], 2).transpose(1, 2), script_probs

def create_biscript_model(config):
    """Crée un modèle YOLOv8 avec tête bi-script"""
    from ultralytics import YOLO
    
    # Charger le modèle de base
    model = YOLO(config['base_model'])
    
    # Remplacer la tête de détection
    model.model.model[-1] = BiScriptDetectionHead(
        nc=config['num_classes'],
        anchors=model.model.model[-1].anchors,
        ch=model.model.model[-1].ch,
        router_channels=256
    )
    
    return model
