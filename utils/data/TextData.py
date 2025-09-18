import os
from typing import Set, Tuple
import numpy as np
import scipy.sparse
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.data import file_utils

# ==============================================================================
# === DATASET AND DATALOADER CLASS ===
# ==============================================================================


class BilingualTextDataset(Dataset):
    def __init__(
        self,
        bow_en,
        bow_cn,
        doc_embeddings_en=None,
        doc_embeddings_cn=None,
        global_bow_en=None,
        global_bow_cn=None,
    ):
        self.bow_en = bow_en
        self.bow_cn = bow_cn
        self.global_bow_en = global_bow_en
        self.global_bow_cn = global_bow_cn
        self.doc_embeddings_en = doc_embeddings_en
        self.doc_embeddings_cn = doc_embeddings_cn
        self.bow_size_en = len(self.bow_en)
        self.bow_size_cn = len(self.bow_cn)

    def __len__(self):
        return max(self.bow_size_en, self.bow_size_cn)

    def __getitem__(self, index):
        en_idx = index % self.bow_size_en
        cn_idx = index % self.bow_size_cn

        return {
            "bow_en": self.bow_en[en_idx],
            "bow_cn": self.bow_cn[cn_idx],
            "global_bow_en": self.global_bow_en[en_idx]
            if self.global_bow_en is not None
            else torch.zeros(1),
            "global_bow_cn": self.global_bow_cn[cn_idx]
            if self.global_bow_cn is not None
            else torch.zeros(1),
            "doc_embedding_en": self.doc_embeddings_en[en_idx]
            if self.doc_embeddings_en is not None
            else torch.zeros(1),
            "doc_embedding_cn": self.doc_embeddings_cn[cn_idx]
            if self.doc_embeddings_cn is not None
            else torch.zeros(1),
        }


# ==============================================================================
# === MAIN HANDLER CLASS ===
# ==============================================================================


class DatasetHandler:
    def __init__(self, dataset, batch_size, lang1, lang2, k_neighbors, device=0):
        data_dir = f"./data/{dataset}"
        self.device = device
        self.batch_size = batch_size
        self.k_neighbors = k_neighbors

        # Read basic data (without global_bow) for both languages
        (
            self.train_texts_en,
            self.test_texts_en,
            self.train_bow_matrix_en,
            self.test_bow_matrix_en,
            self.vocab_en,
            self.doc_embeddings_en,
            self.word_embeddings_en,
        ) = self.read_data(data_dir, lang1)

        (
            self.train_texts_cn,
            self.test_texts_cn,
            self.train_bow_matrix_cn,
            self.test_bow_matrix_cn,
            self.vocab_cn,
            self.doc_embeddings_cn,
            self.word_embeddings_cn,
        ) = self.read_data(data_dir, lang2)

        # Set dimensions
        self.train_size_en = len(self.train_texts_en)
        self.train_size_cn = len(self.train_texts_cn)
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_size_cn = len(self.vocab_cn)

        # --- Dynamic Global BoW Generation ---
        print("\nStarting dynamic Global BoW generation...")

        # Generate for the first language (e.g., 'en')
        self.global_bow_en = self._create_word_level_global_bow(
            bow_anchor=self.train_bow_matrix_en,
            bow_cross=self.train_bow_matrix_cn,
            word_emb_anchor=self.word_embeddings_en,
            word_emb_cross=self.word_embeddings_cn,
            lang_code=lang1,
        )

        # Generate for the second language (e.g., 'cn')
        self.global_bow_cn = self._create_word_level_global_bow(
            bow_anchor=self.train_bow_matrix_cn,
            bow_cross=self.train_bow_matrix_en,
            word_emb_anchor=self.word_embeddings_cn,
            word_emb_cross=self.word_embeddings_en,
            lang_code=lang2,
        )
        print("Global BoW generation complete.\n")

        # Move all necessary data to GPU
        (
            self.doc_embeddings_en,
            self.doc_embeddings_cn,
            self.global_bow_en,
            self.global_bow_cn,
            self.train_bow_matrix_en,
            self.test_bow_matrix_en,
            self.train_bow_matrix_cn,
            self.test_bow_matrix_cn,
        ) = self.move_to_cuda(
            self.doc_embeddings_en,
            self.doc_embeddings_cn,
            self.global_bow_en,
            self.global_bow_cn,
            self.train_bow_matrix_en,
            self.test_bow_matrix_en,
            self.train_bow_matrix_cn,
            self.test_bow_matrix_cn,
        )

        # Create DataLoaders
        self.train_loader = DataLoader(
            BilingualTextDataset(
                self.train_bow_matrix_en,
                self.train_bow_matrix_cn,
                self.doc_embeddings_en,
                self.doc_embeddings_cn,
                self.global_bow_en,
                self.global_bow_cn,
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        self.test_loader = DataLoader(
            BilingualTextDataset(self.test_bow_matrix_en, self.test_bow_matrix_cn),
            batch_size=batch_size,
            shuffle=False,
        )

    def move_to_cuda(self, *arrays):
        results = []
        for arr in arrays:
            if arr is None:
                results.append(None)
                continue

            if isinstance(arr, scipy.sparse.spmatrix):
                tensor = torch.from_numpy(arr.toarray()).float()
            else:
                tensor = (
                    torch.from_numpy(arr).float()
                    if isinstance(arr, np.ndarray)
                    else arr.float()
                )

            if torch.cuda.is_available():
                tensor = tensor.to(f"cuda:{self.device}")
            results.append(tensor)

        return results if len(results) > 1 else results[0]

    def read_data(self, data_dir, lang):
        print(f"Reading data for language: {lang}")
        train_texts = file_utils.read_texts(
            os.path.join(data_dir, f"train_texts_{lang}.txt")
        )
        test_texts = file_utils.read_texts(
            os.path.join(data_dir, f"test_texts_{lang}.txt")
        )
        vocab = file_utils.read_texts(os.path.join(data_dir, f"vocab_{lang}"))

        train_bow_matrix = scipy.sparse.load_npz(
            os.path.join(data_dir, f"train_bow_matrix_{lang}.npz")
        ).toarray()
        test_bow_matrix = scipy.sparse.load_npz(
            os.path.join(data_dir, f"test_bow_matrix_{lang}.npz")
        ).toarray()

        doc_embeddings = np.load(
            os.path.join(data_dir, f"doc_embeddings_{lang}_train.npy")
        )
        word_embeddings = np.load(
            os.path.join(data_dir, f"word_embeddings_{lang}.npy")
        )

        return (
            train_texts,
            test_texts,
            train_bow_matrix,
            test_bow_matrix,
            vocab,
            doc_embeddings,
            word_embeddings,
        )

    def _compute_word_similarities(self, word_embeddings: np.ndarray) -> np.ndarray:
        return cosine_similarity(word_embeddings)

    def _find_top_k_word_neighbors(
        self, similarities: np.ndarray, word_idx: int, k: int, exclude_self: bool = True
    ) -> np.ndarray:
        sim_scores = similarities[word_idx].copy()
        if exclude_self:
            sim_scores[word_idx] = -1

        k = min(k, len(sim_scores) - (1 if exclude_self else 0))
        if k <= 0:
            return np.array([], dtype=int)

        top_k_indices = np.argpartition(sim_scores, -k)[-k:]
        return top_k_indices[np.argsort(sim_scores[top_k_indices])[::-1]]

    def _create_augmented_bow(
        self,
        active_word_indices: Set[int],
        inner_similarities: np.ndarray,
        cross_similarities: np.ndarray,
        vocab_size_inner: int,
        vocab_size_cross: int,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bow_inner_aug = np.zeros(vocab_size_inner)
        bow_cross_aug = np.zeros(vocab_size_cross)

        for word_idx in active_word_indices:
            # Inner neighbors (same language)
            if word_idx < vocab_size_inner:
                inner_neighbors = self._find_top_k_word_neighbors(
                    inner_similarities, word_idx, k, exclude_self=True
                )
                bow_inner_aug[inner_neighbors] += 1
                bow_inner_aug[word_idx] += 1

            # Cross-lingual neighbors (other language)
            if word_idx < cross_similarities.shape[0]:
                cross_neighbors = self._find_top_k_word_neighbors(
                    cross_similarities, word_idx, k, exclude_self=False
                )
                bow_cross_aug[cross_neighbors] += 1

        return bow_inner_aug, bow_cross_aug

    def _create_word_level_global_bow(
        self,
        bow_anchor: np.ndarray,
        bow_cross: np.ndarray,
        word_emb_anchor: np.ndarray,
        word_emb_cross: np.ndarray,
        lang_code: str,
    ) -> np.ndarray:
        n_docs = bow_anchor.shape[0]
        vocab_size_anchor, vocab_size_cross = (
            word_emb_anchor.shape[0],
            word_emb_cross.shape[0],
        )

        print(
            f"--- Processing {lang_code.upper()} documents with word-level neighbors"
            f" (k={self.k_neighbors}) ---"
        )

        print("Calculating word similarity matrices...")
        inner_word_similarities = self._compute_word_similarities(word_emb_anchor)
        cross_word_similarities = cosine_similarity(word_emb_anchor, word_emb_cross)

        all_combined_bows = []

        for i in tqdm(range(n_docs), desc=f"{lang_code.upper()} Processing"):
            active_words = set(np.nonzero(bow_anchor[i])[0])

            if not active_words:
                # The format is always EN | CN
                combined_bow = np.zeros(self.vocab_size_en + self.vocab_size_cn)
            else:
                bow_inner_aug, bow_cross_aug = self._create_augmented_bow(
                    active_words,
                    inner_word_similarities,
                    cross_word_similarities,
                    vocab_size_anchor,
                    vocab_size_cross,
                    self.k_neighbors,
                )

                # Ensure EN | CN format
                if lang_code == "en":  # Assuming 'en' is lang1
                    combined_bow = np.concatenate((bow_inner_aug, bow_cross_aug))
                else:  # Assuming the other lang (e.g., 'cn') is the anchor
                    combined_bow = np.concatenate((bow_cross_aug, bow_inner_aug))

            all_combined_bows.append(combined_bow)

        return np.vstack(all_combined_bows).astype(np.float32)