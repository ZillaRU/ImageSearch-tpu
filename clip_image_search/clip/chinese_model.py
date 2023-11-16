import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .npuengine import EngineOV
import time


class ChineseCLIP():
    def __init__(self,
                 is_processing: bool,
                 batch_size: int=1,
                 embed_dim: int=512,
                 # vision
                 image_resolution: int=224,
                 # text
                 transformer_width: int=512,):
        super().__init__()
        self.is_processing = is_processing
        if not is_processing:
            self.visual = EngineOV(f'./clip_image_search/clip/bmodels/chinese_clip/chinese_clip_imgencoder-1-3-224-224.bmodel')
            self.text_encoder = EngineOV('./clip_image_search/clip/bmodels/chinese_clip/chinese_clip_text_encoder-1_77-1_77.bmodel')
        else:
            self.visual = EngineOV(f'./clip_image_search/clip/bmodels/chinese_clip/chinese_clip_imgencoder-{batch_size}-3-224-224.bmodel')
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

    def encode_image(self, image):
        st_time = time.time()
        img_emb = torch.from_numpy(self.visual([image.numpy().astype(np.float32)])[0])
        if not self.is_processing:
            print('====================== Image Encoding: ', time.time() - st_time)
        return img_emb

    def encode_text(self, text):
        assert not self.is_processing
        st_time = time.time()
        txt_emb = torch.from_numpy(self.text_encoder([text.numpy().astype(np.int32)])[0])
        print('====================== Text Encoding: ', time.time() - st_time)
        return txt_emb

    def _encode(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text