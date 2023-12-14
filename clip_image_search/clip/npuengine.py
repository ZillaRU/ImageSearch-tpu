from tpu_perf.infer import SGInfer
import numpy as np 
import time 
import torch
import os

SGTypeTuple = (
   (np.float32, 0),
   (np.int32, 6),
   (np.uint32, 7),
   (np.int8, 2),
   (np.uint8, 3),
)

typemap = {
    0: np.float32,
    6: np.int32,
    7: np.uint32,
    2: np.int8,
    3: np.uint8,
}


def generate_func(shapes, dtype, mode=1):
    # 0: random 
    # 1: zero
    return np.random.random(shapes).astype(dtype) if mode == 0 else np.zeros(shapes).astype(dtype)


class EngineOV:
    
    def __init__(self, model_path="", batch=1,device_id=0) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            # print(">>>> device_id is in os.environ. and device_id = ",device_id)
        self.model = SGInfer(model_path , batch=batch, devices=[device_id])
    
        
    def __call__(self, args):
        start = time.time()
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        task_id = self.model.put(*values)
        task_id, results, valid = self.model.get()
        return results

