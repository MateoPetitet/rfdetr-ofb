"""
Created on Tue May 27 04:03:52 2025

@author: Matéo Petitet for OFB/Parc Naturel Marin de Martinique
"""
# -*- coding: utf-8 -*-
# Importations
import os
import torch
from rfdetr import RFDETRLarge, RFDETRBase  # Assurez-vous que le package rf-detr est bien installé
import argparse

if __name__ == '__main__' :
    
    #Parser
    parser = argparse.ArgumentParser(description='RF-DETR fine-tuning.')
    parser.add_argument('--path', '--p', type=str, default='data/', help='Path to the data directory ; default : /data/')
    parser.add_argument('--output', '--o', type=str, default='output/', help='Path to the output directory ; default : /output/')
    parser.add_argument('--model_type', '--m', type=str, choices =['base','Base','large','Large'], default='large', help='Base RF-DETR model used ; default : large')
    parser.add_argument('--epochs', '--e', type=int, default=500, help='Numbers of epochs for training ; default : 500')
    parser.add_argument('--learning_rate', '--lr', type=float, default=5e-5, help='Learning rate ; default : 5e-5 (big images, small dataset')
    parser.add_argument('--checkpoint_interval', '--ci', type=int, default=15, help='Number of epochs between each checkpoint ; default : 15')
    parser.add_argument('--resume_path', '--r', type=str, default='checkpoint/', help='Path to the folder containing the checkpoint to resume training from ; default : /checkpoint/')

    args = parser.parse_args()

    # Vérification de la disponibilité du GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device utilisé :', device)

    data_dir = os.path.join(args.path)

    output_path = os.path.join(args.output)

    if args.model_type == 'large' or args.model_type == 'Large':
        model = RFDETRLarge(pretrained=True)
    else:
        model = RFDETRBase(pretrained=True)

    resume_dir = args.resume_path
    if os.path.isdir(resume_dir):
        resume = os.listdir(resume_dir)
        if resume!=[]:
            cp_resume=os.path.join(resume_dir, resume[0])
    else:
        cp_resume=False

    num_epochs = args.epochs

    learn = args.learning_rate

    cp_interval = args.checkpoint_interval

    model.train(dataset_dir=data_dir, num_workers=32, epochs=num_epochs, batch_size=8, grad_accum_steps=2, lr=learn, output_dir=output_path, checkpoint_interval=cp_interval, resume=cp_resume)
