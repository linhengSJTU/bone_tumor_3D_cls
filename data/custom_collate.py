import torch

def CustomCollate(batch):
    """Custom collate fn for dealing with batches of images
    """
    imgs = []
    labels = []
    img_names = []
    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])
        img_names.append(sample[2])
    return torch.stack(imgs, 0), torch.stack(labels, 0), img_names