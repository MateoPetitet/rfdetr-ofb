"""
Created on Wed Apr 30 09:56:00 2025

@author: Matéo Petitet for OFB/Parc Naturel Marin de Martinique
"""
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from rfdetr import  RFDETRBase  # Assurez-vous que le package rf-detr est bien installé
# -*- coding: utf-8 -*-
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
photo_dir = "donnees_test/photos"  # chemin vers les photos
poids = "donnees_test/checkpoints/checkpoint0059.pth"
model = RFDETRBase(pretrain_weights=poids)
print("Device utilisé :", device)
print("Modèle utilisé :", poids)

def draw_boxes(image, boxes, confidences, color="red"):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    font_size = max(16, w//50)
    box_thick = max(2, w//200)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default(font_size)

    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)         #on évite de dépasser négativement
        x2, y2 = min(w, x2,), min(h, y2)        #on évite de dépasser positivement
        
        label = f"fish {conf : .2f}"
        
        # Dessiner la boîte
        for i in range(box_thick):  # Dessin plus épais : on dessine n rectangles
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

        # taille du texte
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # fond et texte du label
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill=color)
        draw.text((x1, y1 - text_height), label, fill="white", font=font)

    return image

files = os.listdir(photo_dir)    # Assume that data structure is folder>files

for image in files:
    image_path = os.path.join(photo_dir, image)
    image_inferee = Image.open(image_path)
    detections = model.predict(image_inferee, threshold=0.3)
    detection_coord = detections.xyxy
    detection_scores = detections.confidence
    detections_image = image_inferee.copy()
    detections_image = draw_boxes(detections_image, detection_coord, detection_scores, color="red")
    detections_image.save(f"donnees_test/photos/{image}_inferee.png")

