import streamlit as st
import os
from PIL import Image
import pickle
import faiss
import numpy as np
import torch
import time
st.set_page_config(layout="wide")

@st.cache_resource
def load_clip(language, device=0):
    model, img_preprocess = clip.load(name=language, processing=False)
    print("============== Models are ready. ==============")
    return model, img_preprocess

@st.cache_resource
def load_db(faiss_index_path, embeddings_path):
    # load faiss index
    index = faiss.read_index(faiss_index_path)
    print("============== Faiss is ready. ==============")
    # load embeddings
    with open(embeddings_path, 'rb') as f:
        results = pickle.load(f)
    embedding_path_list = results['img_path']
    print("============== Embedding dataset is ready. ==============")
    return index, embedding_path_list


if __name__ == '__main__':
    import sys
    lang = sys.argv[1]
    assert lang in ['EN', 'CH']
    import glo_config
    glo_config._init()
    glo_config.set_value('lang', lang)
    import clip_image_search.clip as clip
    st.sidebar.title('ImgSearch with Airbox⚡')

    device = 0
    model, preprocess = load_clip(lang, device)

    GALLERY_COLLECTION = './gallery_collection'
    INDEX_COLLECTION = f'./results/{lang}'
    DUMMY_NEW = '========= New Gallery ========='
    gallery_list = next(os.walk(INDEX_COLLECTION))[1]

    gallery_list.append(DUMMY_NEW)
    # select box
    selected_galley = st.sidebar.selectbox('Choose gallery', gallery_list, index=0)
    if selected_galley != DUMMY_NEW:
        faiss_index_path = f'./results/{lang}/{selected_galley}/index.faiss'
        embeddings_path = f'./results/{lang}/{selected_galley}/embeddings.pkl'
        index, embedding_path_list = load_db(faiss_index_path, embeddings_path)

        search_mode = st.sidebar.selectbox('Search mode', ('Text', 'Upload Image', 'Image'), index=0)

        num_search = st.sidebar.slider('Number of search results', 1, 10, 5)
        images_per_row = st.sidebar.slider('Images per row', 1, num_search, min(5, num_search))
        uploaded_files = st.sidebar.file_uploader("Add images to gallery", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if uploaded_files is not None:
            for file in uploaded_files:
                file_path = os.path.join(GALLERY_COLLECTION, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())
                st.write(f"File saved to {file_path}")
        
        if search_mode == 'Image':
            img_idx = st.slider('Image index', 0, len(embedding_path_list)-1, 0)
            img_path = embedding_path_list[img_idx]
            img = Image.open(img_path).convert('RGB')
            st.image(img, caption=f'Query Image: {img_path}', width=256)
            img_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                features = model.encode_image(img_tensor)
        elif search_mode == 'Upload Image':
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, width=256)
                img_tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    features = model.encode_image(img_tensor)
            else:
                features = None
        else:
            # search by text
            query_text = st.text_input('Enter a search term:')
            if len(query_text.strip()) == 0:
                features = None
            else:
                with torch.no_grad():
                    text = clip.tokenize([query_text])
                    features = model.encode_text(text)
        
        if features is not None:
            features /= features.norm(dim=-1, keepdim=True)
            embedding_query = features.detach().cpu().numpy().astype(np.float32)
            st_time = time.time()
            D,I = index.search(embedding_query, num_search)
            print('====================== Querying: ', time.time() - st_time)
            match_path_list = [embedding_path_list[i] for i in I[0]]

            # calculate number of rows
            num_rows = -(-num_search // images_per_row)  # Equivalent to ceil(num_search / images_per_row)

            # display
            for i in range(num_rows):
                cols = st.columns(images_per_row)
                for j in range(images_per_row):
                    idx = i*images_per_row + j
                    if idx < num_search:
                        path = match_path_list[idx]
                        distance = D[0][idx]
                        img = Image.open(path).convert('RGB')
                        cols[j].image(img, caption=f'Distance: {distance:.2f} path {path}', use_column_width=True)

    else:
        with st.form(key='new_gallery_form'):
            def create_gallery():
                if new_gallery_files is None:
                    st.error('No images selected')
                    return
                if new_gallery_name.strip()!= "" and new_gallery_name != DUMMY_NEW and new_gallery_name not in gallery_list:
                    gallery_path = os.path.join(GALLERY_COLLECTION, new_gallery_name)
                    index_path = os.path.join(INDEX_COLLECTION, new_gallery_name)
                    if os.path.exists(gallery_path):
                        st.error(f"Gallery {new_gallery_name} already exists.")
                    else:
                        os.mkdir(gallery_path)
                        for file in new_gallery_files:
                            file_path = os.path.join(gallery_path, file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.read())
                        st.write(f"Files saved to {gallery_path}")
                    if os.path.exists(index_path):
                        st.error(f"{glo_config.get_value('lang')} Index {new_gallery_name} already exists")
                    else:
                        # import subprocess
                        os.mkdir(index_path)
                        extraction_cmd = f''' python3 ./clip_image_search/extract_embeddings.py \
                                --lang {glo_config.get_value('lang')} \
                                --img_dir {gallery_path} \
                                --save_path {os.path.join(index_path, 'embeddings.pkl')} \
                                --batch_size {batchsize}\
                                --num_workers {num_worker}
                            '''
                        # show logs using streamlit
                        # subprocess.run(extraction_cmd, shell=True)
                        st.write('Loading image encoder and extracting embeddings ...')
                        import time; st_time = time.time()
                        os.system(extraction_cmd)
                        st.write(f'Extraction finished in {time.time() - st_time} sec. \nStart building index...')
                        index_cmd = f''' python3 ./clip_image_search/build_index.py \
                            --embeddings_path {os.path.join(index_path, 'embeddings.pkl')} \
                            --save_path {os.path.join(index_path, 'index.faiss')}
                        '''
                        st_time = time.time()
                        os.system(index_cmd)
                        st.write(f'Buliding index finished in {time.time() - st_time} sec.')
            
            new_gallery_name = st.text_input("New gallery name", value='', key=None)
            new_gallery_files = st.file_uploader("Add images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
            batchsize = st.radio("Batch size", (1, 8), index=1, horizontal=True)
            num_worker = st.slider("Num of workers", 1, 8, 8)
            submit_button = st.form_submit_button(label='Submit', on_click=create_gallery)