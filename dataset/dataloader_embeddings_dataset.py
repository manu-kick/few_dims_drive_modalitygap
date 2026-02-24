# We need to create a custom Dataset that loads the precomputed text embeddings from disk and returns them in the __getitem__ method. 
# We have to ensure that for each image, we return a random caption embedding from the 5 available, and that we don't repeat the same caption for the same image across epochs.
# we Need to get all the samples from the precomputed .npz files, and then create a Dataset that can be used with a DataLoader to feed the model during training.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader    

class EmbeddingsDataset(Dataset):
    def __init__(self, precomputed_dir, split_name="flickr30k"):
        """
        Args:
            precomputed_dir (str): Directory where the precomputed .npz files are stored.
            split_name (str): Name of the dataset split (e.g., "flickr30k") to filter the files.
        """
        self.text_embeddings = []
        self.vision_embeddings = []
        
        # Load all .npz files that match the split_name
        for fn in os.listdir(precomputed_dir):
            if fn.endswith(".npz") and split_name in fn:
                data = np.load(os.path.join(precomputed_dir, fn))
                self.text_embeddings.append(data["text_embeddings"])
                self.vision_embeddings.append(data["vision_embeddings"])
        
        # Concatenate all loaded embeddings into single arrays
        self.text_embeddings = np.concatenate(self.text_embeddings, axis=0)  # [N, D]
        self.vision_embeddings = np.concatenate(self.vision_embeddings, axis=0)  # [N, D]

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        text_emb = torch.as_tensor(self.text_embeddings[idx]).float()
        vision_emb = torch.as_tensor(self.vision_embeddings[idx]).float()
        return text_emb, vision_emb
    
def get_embeddings_dataloaders(precomputed_dir, split_name="flickr30k", batch_size=32, shuffle=True):
    dataset = EmbeddingsDataset(precomputed_dir, split_name)
    # we 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader