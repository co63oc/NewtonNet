import torch
import numpy as np
import paddle

a = torch.load("newtonnet.pt")
for k, item in a.items():
    # a[k] = paddle.to_tensor(item.cpu().numpy()).to(dtype="float32")
    if "node_embed" in k:
        print(item)
    a[k] = paddle.to_tensor(item.cpu().numpy())
# paddle.save(a, "newtonnet_float32.pdparams")
paddle.save(a, "newtonnet.pdparams")


