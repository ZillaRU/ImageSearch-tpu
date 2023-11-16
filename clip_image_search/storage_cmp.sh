#!/usr/bin/env bash

dataset=$1
compression_type=$2

compressed_folder="./gallery_collection/$dataset/"
file_path="./results/$compression_type/$dataset/embeddings.pkl"

file_size_h=$(du -b -h "$file_path" | awk '{print $1}')
folder_size_h=$(du -b -s -h "$compressed_folder" | awk '{print $1}')
echo "图片文件夹大小：$folder_size_h"
echo "Embedding大小：$file_size_h"

file_size=$(du -b "$file_path" | awk '{print $1}')
folder_size=$(du -b -s "$compressed_folder" | awk '{print $1}')
ratio=$(awk "BEGIN {printf \"%.2f\",  $folder_size/$file_size}")

echo "压缩比：$ratio"