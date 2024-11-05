import os
import random
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def set_seed(config = None):
    """
    Set seed for reproducibility
    """
    if config is None or "seed" not in config:
        raise ValueError("Seed can not be set")
    
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore

def tsne_visualization(last_hidden, labels, epoch, exp_folder, type='test', plot_title="t-SNE Visualization", seed=42):
    """
    Visualize the last hidden layer using t-SNE
    """
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300, random_state=seed)
    tsne_results = tsne.fit_transform(last_hidden)

    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Labels")
    plt.title(f"{plot_title} - Epoch {epoch+1}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()
    plt.savefig(f"{exp_folder}/tsne/tsne_epoch_{type}_{epoch+1}.png")