import os
import yaml
import scipy.io
from scipy.io import loadmat
from runners.Runner import Runner
import argparse

from utils.data import file_utils
from utils.data.TextData import DatasetHandler
from utils import miscellaneous, seed
from utils.TU import *
from utils.eval import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--weight_kl_loss', type=float, default=50.0, help='Weight for KL loss between global and local theta distributions')
    parser.add_argument('--weight_cka_loss', type=float, default=0.1, help='Weight for CKA loss between local theta and document embeddings')
    parser.add_argument('--k_neighbors', type=int, default=10)
    parser.add_argument('--device', type=int, default=0, help='CUDA device index to use')
    args = parser.parse_args()
    return args

def export_beta(beta, vocab, output_prefix, lang):
    num_top_word = 15
    topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=num_top_word)
    file_utils.save_text(topic_str_list, path=f'{output_prefix}/T{num_top_word}_{lang}.txt')
    return topic_str_list

RESULT_DIR= 'output'
def main():
    args = parse_args()

    args = file_utils.update_args(args, f'./configs/model/{args.model}.yaml')
    args = file_utils.update_args(args, f'./configs/dataset/{args.dataset}.yaml')

    current_time = miscellaneous.get_current_datetime()
    output_prefix = os.path.join(RESULT_DIR + "/" + str(args.model) + "/" +str(args.dataset), current_time)
    
    miscellaneous.create_folder_if_not_exist(output_prefix)
    seed.seedEverything(args.seed)

    print('\n' + yaml.dump(vars(args), default_flow_style=False))
    
    dataset_handler = DatasetHandler(args.dataset, args.batch_size, args.lang1, args.lang2,  k_neighbors=args.k_neighbors,device=args.device)

    args.doc_embeddings_en = dataset_handler.doc_embeddings_en
    args.doc_embeddings_cn = dataset_handler.doc_embeddings_cn

    args.vocab_size_en = len(dataset_handler.vocab_en)
    args.vocab_size_cn = len(dataset_handler.vocab_cn)
    args.vocab_en = dataset_handler.vocab_en
    args.vocab_cn = dataset_handler.vocab_cn
    
    # Add word similarity and alignment mask to args

    runner = Runner(args)

    beta_en, beta_cn = runner.train(dataset_handler.train_loader)

    topic_str_list_en = export_beta(beta_en, dataset_handler.vocab_en, output_prefix, lang=args.lang1)
    topic_str_list_cn = export_beta(beta_cn, dataset_handler.vocab_cn, output_prefix, lang=args.lang2)

    for i in range(len(topic_str_list_en)):
        print(topic_str_list_en[i])
        print(topic_str_list_cn[i])

    train_theta_en, train_theta_cn = runner.test(dataset_handler.train_loader.dataset)
    test_theta_en, test_theta_cn = runner.test(dataset_handler.test_loader.dataset)

    rst_dict = {
        'beta_en': beta_en,
        'beta_cn': beta_cn,
        'train_theta_en': train_theta_en,
        'train_theta_cn': train_theta_cn,
        'test_theta_en': test_theta_en,
        'test_theta_cn': test_theta_cn,
    }

    scipy.io.savemat(f'{output_prefix}/rst.mat', rst_dict)
    
    #Calculate TU
    num_top_word = 15
    texts = list()
    with open(f'{output_prefix}/T{num_top_word}_{args.lang1}.txt', 'r') as file:
        for line in file:
            texts.append(line.strip())

    TU = TU_eva(texts)
    print(f"TU_{args.lang1}: {TU:.5f}")
    
    texts = list()
    with open(f'{output_prefix}/T{num_top_word}_{args.lang2}.txt', 'r') as file:
        for line in file:
            texts.append(line.strip())

    TU = TU_eva(texts)
    print(f"TU_{args.lang2}: {TU:.5f}")
    
    #----------Eval theta and more--------------
    dataset_name = args.dataset
    model_name = args.model
    num_topics = args.num_topic

    base_data_dir = f"data/{dataset_name}"
    mat_path = f'{output_prefix}/rst.mat'

    en_top_words_path = f'{output_prefix}/T{num_top_word}_{args.lang1}.txt'
    cn_top_words_path = f'{output_prefix}/T{num_top_word}_{args.lang2}.txt'

    en_corpus_path = f"{base_data_dir}/train_texts_en.txt" 
    train_labels_en_path = f"{base_data_dir}/train_labels_en.txt"
    train_labels_cn_path = f"{base_data_dir}/train_labels_cn.txt"
    test_labels_en_path = f"{base_data_dir}/test_labels_en.txt"
    test_labels_cn_path = f"{base_data_dir}/test_labels_cn.txt"
    if args.lang2 == "ja":
        train_labels_cn_path = f"{base_data_dir}/train_labels_ja.txt"
        test_labels_cn_path = f"{base_data_dir}/test_labels_ja.txt"

    print(f"--- Evaluating Model: {model_name}, Dataset: {dataset_name}, K={num_topics} ---")

    print("\n--- Loading Data ---")
    train_labels_en = load_labels_txt(train_labels_en_path)
    train_labels_cn = load_labels_txt(train_labels_cn_path)
    test_labels_en = load_labels_txt(test_labels_en_path)
    test_labels_cn = load_labels_txt(test_labels_cn_path)
    if any(arr.size == 0 for arr in [train_labels_en, train_labels_cn, test_labels_en, test_labels_cn]):
        print("Error: Failed to load one or more label files. Exiting.")
        exit()
    print("Labels loaded successfully.")

    try:
        mat = loadmat(mat_path)
        train_theta_en = mat["train_theta_en"]
        train_theta_cn = mat["train_theta_cn"]
        test_theta_en = mat["test_theta_en"]    
        test_theta_cn = mat["test_theta_cn"]
        print(f"Results matrix loaded successfully from {mat_path}.")
    except FileNotFoundError:
        print(f"Error: Results matrix file not found at {mat_path}. Exiting.")
        exit()
    except KeyError as e:
        print(f"Error: Key {e} not found in results matrix {mat_path}. Exiting.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading the .mat file: {e}. Exiting.")
        exit()

    print("Loading text data for Coherence/Diversity...")
    en_top_words_list = split_text_word(en_top_words_path)
    cn_top_words_list = split_text_word(cn_top_words_path)

    print("\n================= Classification =================")
    cls_results = crosslingual_cls(
        train_theta_en, train_theta_cn,
        test_theta_en, test_theta_cn,
        train_labels_en, train_labels_cn,
        test_labels_en, test_labels_cn
    )
    print_results(cls_results)

if __name__ == '__main__':
    main()