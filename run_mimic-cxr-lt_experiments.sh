#!/bin/bash

echo "Training CE on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ce \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training class-balanced CE on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ce \
                     --rw_method cb \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training sklearn re-weighted CE on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ce \
                     --rw_method sklearn \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training focal loss on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss focal \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training class-balanced focal loss on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss focal \
                     --rw_method cb \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training sklearn re-weighted focal loss on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss focal \
                     --rw_method sklearn \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training LDAM loss on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ldam \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training class-balanced LDAM loss on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ldam \
                     --rw_method cb \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training class-balanced LDAM loss w/ DRW on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ldam \
                     --rw_method cb \
                     --drw \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training sklearn re-weighted LDAM loss on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ldam \
                     --rw_method sklearn \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training sklearn re-weighted LDAM loss w/ DRW on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ldam \
                     --rw_method sklearn \
                     --drw \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training mixup on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ce \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \
                     --mixup \
                     --mixup_alpha 0.2 \

echo "Training balanced mixup on mimic-cxr-lt..."
python src/main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ce \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \
                     --bal_mixup \
                     --mixup_alpha 0.2 \

