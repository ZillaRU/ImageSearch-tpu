# Introduction

This code implements a versatile image search engine leveraging the CLIP model and FAISS, capable of processing both text-to-image and image-to-image queries.

# Install Environment
```sh
virtualenv clipfaiss
source ./clipfaiss/bin/activate
pip install -r requirements.txt
```

# Download bmodels for running CLIP on TPU
```
python -m pip install dfn
# download CLIP VIT-b32 and put these files into ./clip_image_search/clip/bmodels/EN
python3 -m dfn --url https://disk.sophgo.vip/sharing/optDG3uDs
# download ChineseCLIP VIT-16 and put these files into ./clip_image_search/clip/bmodels/CH
python3 -m dfn --url https://disk.sophgo.vip/sharing/qw6hvmVWs
```


# Demo
You can initialize your own gallery and search images easily.
```sh
streamlit run app.py CH # for ChineseCLIP VIT-B16
streamlit run app.py EN # for CLIP-VIT-B32 
```


# Preprocess manually
```python
python clip_image_search/extract_embeddings.py --img_dir your_dataset_dir --save_path results/embeddings.pkl

python clip_image_search/build_index.py --embeddings_path results/embeddings.pkl --save_path results/index.faiss
```


# Check the compression ratio
You can compare the storage size of the original image dataset and that of its embedding file by the following command.
```sh
bash ./clip_image_search/storage_cmp.sh [The name of gallery] [EN or CH]
```
# Clean the gallery
```sh
rm -rf ./gallery_collection/*
rm -rf results/CH/*
rm -rf results/EN/*
```
# ⚠The number of workers for preprocessing during feature extraction is set to 8 by default.
If your device memory is less than 1G, please set the number of workers to a smaller value to prevent crashing.
