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
def compute_embeddings(language, img_dir, save_path, batch_size, num_workers, device='tpu'):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if language == 'both':
        _langs = ['CH', 'EN']
    else:
        _langs = [language]
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


if __name__ == "__main__":
    import sys
    compute_embeddings()
