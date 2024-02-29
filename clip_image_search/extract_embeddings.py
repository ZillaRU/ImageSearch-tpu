import clip
from search_dataset import ClipSearchDataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch
import click

@click.command()
@click.option('--language', default='EN')
@click.option('--img_dir', default='/data/clip-image-search/gallery_collection/test', help='Directory of images.')
@click.option('--save_path', default='results/EN/test/embeddings.pkl', help='Path to save the embeddings.')
@click.option('--batch_size', default=1, help='Batch size for DataLoader.')
@click.option('--num_workers', default=1, help='Number of workers for DataLoader.')
@click.option('--mode', default='create', help='select embedding mode None or update,')
@click.option('--update_dir', default=None, help='if mode is update, need provide update dir')
def compute_embeddings(language, img_dir, save_path, batch_size, num_workers, mode, update_dir, device='tpu'):
    if not os.path.exists(os.path.dirname(save_path)) and mode is None:
        os.makedirs(os.path.dirname(save_path))
    if language == 'both':
        _langs = ['CH', 'EN']
    else:
        _langs = [language]
    if mode == 'create':
        for lang in _langs:
            model, preprocess = clip.load(lang, device, batch_size=batch_size, processing=True)
            dataset = ClipSearchDataset(img_dir = img_dir, preprocess = preprocess)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            img_path_list, embedding_list = [], []
            for img, img_path in tqdm(dataloader):
                with torch.no_grad():
                    features = model.encode_image(img)
                    features /= features.norm(dim=-1, keepdim=True)
                    embedding_list.extend(features.detach().cpu().numpy())
                    img_path_list.extend(img_path)

            result = {'img_path': img_path_list, 'embedding': embedding_list}
            with open(save_path.replace('/both/', f'/{lang}/'), 'wb') as f:
                pickle.dump(result, f, protocol=4)

    elif mode == "update" and update_dir is not None:
        update_list = [os.path.join(img_dir, file) for file in os.listdir(update_dir)]
        for lang in _langs:
            model, preprocess = clip.load(lang, device, batch_size=batch_size, processing=True)
            dataset = ClipSearchDataset(img_dir=img_dir, preprocess=preprocess, mode='update', update_list=update_list)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            with open(save_path, 'rb') as f:
                result = pickle.load(f)

            for img, img_path in tqdm(dataloader):
                with torch.no_grad():
                    features = model.encode_image(img)
                    features /= features.norm(dim=-1, keepdim=True)
                    result['embedding'].extend(features.detach().cpu().numpy())
                    result['img_path'].extend(img_path)

            with open(save_path.replace('/both/', f'/{lang}/'), 'wb') as f:
                pickle.dump(result, f, protocol=4)




if __name__ == "__main__":
    import sys
    compute_embeddings()
