"""
Created on Wed Apr 30 09:56:00 2025

@author: Matéo Petitet for OFB/Parc Naturel Marin de Martinique
"""
# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from rfdetr import  RFDETRBase
import argparse


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

def inference(img, seuil):
    detections = model.predict(img, threshold=seuil)
    detect_coord = detections.xyxy
    detect_scores = detections.confidence
    return detect_coord, detect_scores

def decoupe(image, img_name, boxes, path):
    save_path = os.path.join(path, img_name)
    os.mkdir(save_path)
    w, h = image.size
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1 = int(max(0, x1)), int(max(0, y1))         #on évite de dépasser négativement
        x2, y2 = int(min(w, x2,)), int(min(h, y2))        #on évite de dépasser positivement
        crop_image = image.copy().crop((x1, y1, x2, y2))
        save_name=f"{x1}-{y1}-{x2}-{y2}.jpg"
        crop_image.save(os.path.join(save_path, save_name))


if __name__ == '__main__':
#Parser
    parser = argparse.ArgumentParser(description='Inference on an existing picture with a specified model.')
    parser.add_argument('--img_path', '--i',  type=str, help='Path to the pictures directory')
    parser.add_argument('--model_path', '--m', type=str, help='Path to the model checkpoint file')
    parser.add_argument('--threshold', '--t',  type=float, default=0.3, help='Threshold value for inference ; default : 0.3')
    parser.add_argument('--visualize', '--v', type=int, choices=[0, 1], default=1, help='1 : save the original pictures with the detection (default), 0 not to do it')
    parser.add_argument('--crop_mode', '--c', type=int, choices=[0, 1], default=1, help='1 : extract the detections from the original picture (default), 0 to not do it.')

    args = parser.parse_args()

    #init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    photo_dir = os.path.abspath(args.img_path)  # chemin vers les photos
    poids = os.path.abspath (args.model_path)
    model = RFDETRBase(pretrain_weights=poids)
    print("Device utilisé :", device)
    print("Modèle utilisé :", poids)
    
    files = os.listdir(photo_dir)    # Assume that data structure is folder>files
    
    #dossier de sauvegarde des photos annotées
    save_path = os.path.join(photo_dir, "inference")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for image in files:
        image_path = os.path.join(photo_dir, image)
        if os.path.isdir(image_path):
            continue        #rien ne sert d'essayer d'inférer un dossier
        image_inferee = Image.open(image_path)
        seuil=args.threshold
        if seuil<0 or seuil>1:
            print ('Please use a threshold between 0 and 1.')
            break
        detection_coord, detection_scores = inference(image_path, seuil)
        if len(detection_scores)==0:    #pas de detection
            continue    #rien ne sert de créer un dossier vide
        image_name, ext = os.path.splitext(image)
        if args.visualize==1:
            detections_image = image_inferee.copy()
            detections_image = draw_boxes(detections_image, detection_coord, detection_scores, color="red")
            save_name = f"{image_name}_inferee.png"
            detections_image.save(os.path.join(save_path, save_name))
        if args.crop_mode==1:
            decoupe(image_inferee, image_name, detection_coord, save_path)

    print(f"Results saved in {save_path}")


