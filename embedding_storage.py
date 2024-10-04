import numpy as np
import os
from typing import Dict, List, Optional

class EmbeddingStorage:
    def __init__(self, storage_dir: str = "embeddings"):
        self.storage_dir = storage_dir
        self.embeddings: Dict[str, np.ndarray] = {}
        self.texts: List[str] = []
        os.makedirs(storage_dir, exist_ok=True)
        self.load_embeddings()

    def add_embedding(self, text: str, embedding: np.ndarray):
        self.embeddings[text] = embedding
        if text not in self.texts:
            self.texts.append(text)
        self.save_embeddings()

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        return self.embeddings.get(text)

    def save_embeddings(self):
        np.savez(os.path.join(self.storage_dir, "embeddings.npz"), **self.embeddings)
        with open(os.path.join(self.storage_dir, "texts.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.texts))

    def load_embeddings(self):
        if os.path.exists(os.path.join(self.storage_dir, "embeddings.npz")):
            loaded = np.load(os.path.join(self.storage_dir, "embeddings.npz"))
            self.embeddings = {k: loaded[k] for k in loaded.files}
        if os.path.exists(os.path.join(self.storage_dir, "texts.txt")):
            with open(os.path.join(self.storage_dir, "texts.txt"), "r", encoding="utf-8") as f:
                self.texts = f.read().splitlines()

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        return self.embeddings

    def get_all_texts(self) -> List[str]:
        return self.texts