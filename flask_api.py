"""
create a flask app, and define the route and function for the following features:
1. create a new gallery
2. search in a specific gallery by text
3. search in a specific gallery by image
"""
from flask import Flask, request, jsonify
import clip_image_search.clip as clip
import os
import faiss
import base64
import pickle
import io
import numpy as np
from PIL import Image


app = Flask(__name__)

def get_base64(img_path):
    try:
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except:
        print(f"Image {img_path} not found.")
        encoded_string = None
    return encoded_string


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


# load model and default gallery before the first request
@app.before_first_request
def load_clip_and_example_gallery():
    print("============== start loading models. ==============")
    en_model, preprocess = clip.load(name='EN', processing=False)
    # ch_model, preprocess = clip.load(name='CH', processing=False)
    app.config['en_model'] = en_model
    # app.config['ch_model'] = ch_model
    app.config['preprocess'] = preprocess
    print("============== Models are ready. ==============")

    if EXAMPLE_GALLERY not in next(os.walk(GALLERY_COLLECTION))[1]:
        print('⚠⚠⚠⚠⚠⚠⚠⚠ No default gallery found. Please create one. ⚠⚠⚠⚠⚠⚠⚠⚠')
        return

    en_faiss_index_path = f'./results/EN/{EXAMPLE_GALLERY}/index.faiss'
    en_embeddings_path = f'./results/EN/{EXAMPLE_GALLERY}/embeddings.pkl'
    en_index, en_embedding_path_list = load_db(en_faiss_index_path, en_embeddings_path)
    # ch_faiss_index_path = f'./results/CH/{EXAMPLE_GALLERY}/index.faiss'
    # ch_embeddings_path = f'./results/CH/{EXAMPLE_GALLERY}/embeddings.pkl'
    # ch_index, ch_embedding_path_list = load_db(ch_faiss_index_path, ch_embeddings_path)
    app.config['active_gallary_name'] = EXAMPLE_GALLERY
    app.config['en_index'] = en_index
    app.config['en_embedding_path_list'] = en_embedding_path_list
    # app.config['ch_index'] = ch_index
    # app.config['ch_embedding_path_list'] = ch_embedding_path_list
    print(f"============== Embedding dataset of {EXAMPLE_GALLERY} is ready. ==============")


@app.route("/")
def index():
    return "ImgSearch with Airbox⚡"

"""
新建图库 【此接口备用】

示例请求：
```json
{
    "gallery_name": "my_new_gallery",
    "images": [
        "img base64 string1",
        "img base64 string2",
        ...
    ]
}
```

建议不要把自建图库的功能开放给体验的用户。
建议预制一个图库和它的中英文embedding和index（没必要调用这个接口，`streamlit run app.py CH` / `streamlit run app.py EN`，即可在图形化界面新建图库）。
"""
@app.route("/create_gallery", methods=["POST"])
def create_gallery():
    data = request.get_json()
    # get the gallery name from the request
    gallery_name = data.get("gallery_name")
    # get the images from the request
    images = data.get("images")
    # get language
    lang = data.get("lang", 'both')
    assert lang in ['CH', 'EN', 'both']

    # save the images
    gallery_path = os.path.join(GALLERY_COLLECTION, gallery_name)
    
    if os.path.exists(gallery_path):
        res = {
            'message': 'The gallery name already exists.',
        }
        return jsonify(res)
    
    os.mkdir(gallery_path)

    for image in images:
        # decode the base64 image
        image = base64.b64decode(image)
        # save the image
        image_path = os.path.join(gallery_path, image.filename)
        with open(image_path, 'wb') as f:
            f.write(image)
    print(f"Images saved to {gallery_path}")
    
    # extract features
    extraction_cmd = f''' python3 ./clip_image_search/extract_embeddings.py \
            --language {lang} \
            --img_dir {gallery_path} \
            --save_path {os.path.join(INDEX_COLLECTION, lang, gallery_name, 'embeddings.pkl')} \
            --batch_size 8\
            --num_workers 8
        '''
    os.system(extraction_cmd)
    
    # build faiss index
    index_cmd = f''' python3 ./clip_image_search/build_index.py \
        --embeddings_path {os.path.join(index_path, 'embeddings.pkl')} \
        --save_path {os.path.join(index_path, 'index.faiss')}
    '''
    st_time = time.time()
    os.system(index_cmd)
    st.write(f'Buliding index finished in {time.time() - st_time} sec.')
    # return a response with a success message in json
    res = {
        "message": f"Gallery created successfully"
    }
    return jsonify(res)

"""
top_n：返回图库中最符合查询文本的前top_n张图。
这个值不要太大。因为图库存储在云端，需要把查询到图片的base64返回过来。
这个值设置的太大，会导致response慢体验差，而实际查询其实是微妙级的。
另外图库里的单张图也不要太大。
示例请求：
1. 
```json
{
    "query_text": "猫趴在草丛边",
    "language": "CH"
}
```
2. 
```json
{
    "query_text": "mountain view, sunset",
    "language": "EN"
}

返回top_n个图片base64和对应的相似度值。

"""
@app.route("/search_by_text", methods=["POST"])
def search_by_text():
    data = request.get_json()
    # get the query text from the request
    query_text = data.get("query_text")
    # get the number of search results
    num_search = int(data.get("top_n", 3)) 
    # get the gallery name from the request
    gallery_name = data.get("gallery_name", "examples") # 建议不要传这个参数，直接默认去从预制的图库查询；不允许用户去切换图库

    if gallery_name != app.config['active_gallary_name']:
        res = {
            'Message': 'The gallery is not avaliable.', # 不允许用户去切换图库
        }
        return jsonify(res)

    # get language
    language = data.get("language")

    if language == 'CH':
        embedding_path_list = app.config['ch_embedding_path_list']
        model = app.config['ch_model']
        index = app.config['ch_index']

    elif language == 'EN':
        embedding_path_list = app.config['en_embedding_path_list']
        model = app.config['en_model']
        index = app.config['en_index']
    else:
        raise ValueError(f"Invalid language: {language}")
    
    # search in the gallery
    tokens = clip.ch_tokenize([query_text]) if language == 'CH' else clip.en_tokenize([query_text])

    text_feature = model.encode_text(tokens)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    embedding_query = text_feature.detach().cpu().numpy().astype(np.float32)
    D, I = index.search(embedding_query, num_search)

    img_base64_list = []
    distance_list = []
    for i in range(num_search):
        temp = get_base64(embedding_path_list[I[0][i]])
        if temp is not None:
            img_base64_list.append(temp)
            distance_list.append(D[0][i].item())
    res = {
        'img_base64_list': img_base64_list,
        'distance_list': distance_list
    }
    return jsonify(res)


"""
top_n：返回图库中和查询图像最相似的前top_n张图。
这个值不要太大。因为图库存储在云端，需要把查询到图片的base64返回过来。
这个值设置的太大，会导致response慢体验差，而实际查询其实是微妙级的。
另外图库里的单张图也不要太大。
示例请求：
```json
{
    "query_image": "img_base64_string"
}
```
返回top_n个图片base64和对应的相似度值。

"""
@app.route("/search_by_image", methods=["POST"])
def search_by_image():
    data = request.get_json()
    # get the query image from the request
    query_image = data.get("query_image")
    # get the number of search results
    num_search = int(data.get("top_n", 3)) 
    # get the gallery name from the request
    gallery_name = data.get("gallery_name", "examples") # 建议不要传这个参数，直接默认去从预制的图库查询；不允许用户去切换图库

    if gallery_name != app.config['active_gallary_name']:
        res = {
            'Message': 'The gallery is not avaliable.', # 不允许用户去切换图库
        }
        return jsonify(res)
    
    model = app.config['en_model']
    index = app.config['en_index']
    embedding_path_list = app.config['en_embedding_path_list']

    # search in the gallery
    img_feature = model.encode_image(app.config['preprocess'](Image.open(io.BytesIO(base64.b64decode(query_image)))).unsqueeze(0))
    D, I = index.search(img_feature, num_search)

    img_base64_list = []
    distance_list = []
    for i in range(num_search):
        temp = get_base64(embedding_path_list[I[0][i]])
        if temp is not None:
            img_base64_list.append(temp)
            distance_list.append(D[0][i].item())
    res = {
        'img_base64_list': img_base64_list,
        'distance_list': distance_list
    }
    return jsonify(res)


if __name__ == '__main__':
    GALLERY_COLLECTION = './gallery_collection'
    INDEX_COLLECTION = f'./results/'
    EXAMPLE_GALLERY = 'examples'
    # use 0.0.0.0 as host
    app.run(debug=False, port=8787, host="0.0.0.0", threaded=False)
