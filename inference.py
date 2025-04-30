"""
Created on Wed Apr 30 09:56:00 2025

@author: Matéo Petitet for OFB/Parc Naturel Marin de Martinique
"""
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import matplotlib.pyplot as plt
import pycocotools.coco as coco
from PIL import Image
from rfdetr import RFDETRLarge, RFDETRBase  # Assurez-vous que le package rf-detr est bien installé
# -*- coding: utf-8 -*-
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
photo_dir = "path/to/pictures"  # chemin vers les photos
poids = "path/to/pth.pth"
model = RFDETRBase(pretrain_weights=poids)
print("Device utilisé :", device)
print("Modèle utilisé :", poids)