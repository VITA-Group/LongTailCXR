#!/bin/bash

echo "Training CE on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ce \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training class-balanced CE on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ce \
                     --rw_method cb \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training sklearn re-weighted CE on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ce \
                     --rw_method sklearn \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training focal loss on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss focal \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training class-balanced focal loss on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss focal \
                     --rw_method cb \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training sklearn re-weighted focal loss on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss focal \
                     --rw_method sklearn \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training LDAM loss on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ldam \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training class-balanced LDAM loss on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ldam \
                     --rw_method cb \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training class-balanced LDAM loss w/ DRW on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ldam \
                     --rw_method cb \
                     --drw \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training sklearn re-weighted LDAM loss on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ldam \
                     --rw_method sklearn \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training sklearn re-weighted LDAM loss w/ DRW on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ldam \
                     --rw_method sklearn \
                     --drw \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \

echo "Training mixup on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ce \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \
                     --mixup \
                     --mixup_alpha 0.2 \

echo "Training balanced mixup on NIH-LT..."
python src/main.py --data_dir /ssd1/greg/NIH_CXR/images \
                     --out_dir nih_results \
                     --dataset nih-cxr-lt \
                     --loss ce \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \
                     --bal_mixup \
                     --mixup_alpha 0.2 \

