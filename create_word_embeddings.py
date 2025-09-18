import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# Use uppercase for constants that do not change during runtime.
BASE_DATA_DIR = 'data'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
DATASETS = [
    {"name": "Amazon_Review", "langs": ["en", "cn"]},
    {"name": "ECNews", "langs": ["en", "cn"]},
    {"name": "Rakuten_Amazon", "langs": ["en", "ja"]}
]
BATCH_SIZE = 128

def load_words_from_file(filepath: str) -> list[str] | None:
    """
    Reads words from a file, skipping empty lines.

    Args:
        filepath: The path to the vocabulary file.

    Returns:
        A list of words, or None if an error occurs.
    """
    if not os.path.exists(filepath):
        print(f"    -> Error: Vocabulary file not found at '{filepath}'")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        
        if not words:
            print(f"    -> Note: Vocabulary file is empty at '{filepath}'.")
            return None
        
        return words
    except Exception as e:
        print(f"    -> Error reading file '{filepath}': {e}")
        return None

def process_and_save_embeddings(model: SentenceTransformer, dataset_name: str, lang: str, base_dir: str):
    """
    Processes a single (dataset, language) pair: loads vocab, 
    creates embeddings, and saves them to a .npy file.
    """
    print(f"\n▶️  Processing: Dataset '{dataset_name}' - Language '{lang}'")
    
    # Dynamically construct file paths
    dataset_dir = os.path.join(base_dir, dataset_name)
    vocab_filepath = os.path.join(dataset_dir, f'vocab_{lang}')
    output_filepath = os.path.join(dataset_dir, f'word_embeddings_{lang}.npy')

    # Load words from the vocabulary file
    words = load_words_from_file(vocab_filepath)
    if not words:
        print("   Skipping due to missing or empty vocabulary file.")
        return

    # Generate embeddings for the words
    print(f"   Generating embeddings for {len(words)} words...")
    start_time = time.time()
    embeddings = model.encode(words, show_progress_bar=True, batch_size=BATCH_SIZE)
    duration = time.time() - start_time
    print(f"   Completed in {duration:.2f} seconds.")

    # Save the embeddings to a .npy file
    try:
        # Ensure the target directory exists
        os.makedirs(dataset_dir, exist_ok=True)
        np.save(output_filepath, embeddings)
        print(f"   💾 Successfully saved to: {output_filepath}")
        print(f"   Shape of saved embeddings: {embeddings.shape}")
    except Exception as e:
        print(f"   -> Error saving file to '{output_filepath}': {e}")

def main():
    """
    Main function: Loads the model and orchestrates the embedding 
    creation process for all specified datasets.
    """
    print(f"--- Starting Embedding Generation Process ---")
    print(f"Using model: {MODEL_NAME}")
    
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Fatal Error: Could not load model '{MODEL_NAME}'. Aborting. Error: {e}")
        return

    for dataset in DATASETS:
        for lang in dataset["langs"]:
            process_and_save_embeddings(
                model=model,
                dataset_name=dataset['name'],
                lang=lang,
                base_dir=BASE_DATA_DIR
            )
            
    print("\n--- ✅ All tasks completed! ---")

if __name__ == '__main__':
    main()