#!/bin/bash

echo "====================="
echo "Create data directory"
mkdir data
echo "====================="
echo "Into data directory"
cd data

echo "====================="
echo "Download Figaro1K dataset from google drive"
gdown https://drive.google.com/uc?id=1MTvPCpkzF0Gb5JhFdxRlDUwu6Cll2gdS

echo "====================="
echo "Decompression dataset.tar"
tar zxvf dataset.tar

echo "====================="
echo "remove unnecessary file"
rm dataset_figaro1k/training/images/._Frame*
rm dataset_figaro1k/training/masks/._Frame*
rm dataset_figaro1k/testing/images/._Frame*
rm dataset_figaro1k/testing/masks/._Frame*

echo "====================="
echo "complete!!"
echo "====================="
