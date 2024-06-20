# models/vit.py
from vit_keras import vit

def vit_model(input_shape):
    return vit.vit_b16(
        image_size=224,
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )
