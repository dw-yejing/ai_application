import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([ T.ToTensor()])
height_width = 32
cifar10_test = torchvision.datasets.CIFAR10(root='F:/dataset', transform=transform, train=False, download=True )
def show_collage(examples):
    box_size = height_width + 2
    num_cols = len(examples)

    collage = Image.new(
        mode="RGB",
        size=(num_cols * box_size,  box_size),
        color=(255, 255, 255),
    )
    for col_idx in range(num_cols):
        array = examples[col_idx][0].permute(1,2,0).numpy()
        array = np.array(array * 255).astype(np.uint8) 
        collage.paste(
            Image.fromarray(array), (col_idx * box_size,  0)
        )

    collage.save('red_image.png')
    return collage

sample_idxs = np.array([6667, 6667, 310, 7426, 5245, 6110])
examples = [cifar10_test[i-1] for i in sample_idxs]
show_collage(examples)