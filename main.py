import os
from dataset_loader import CounterfactDataset, ZsreDataset
from knowledge_editor import KnowledgeEditor, DataEncoder

def generate_cache_if_not_exist(dataset_path: str,
                                cache_prefix: str,
                                prefix_dir: str,
                                src_lang: str,
                                tgt_lang: str,
                                DatasetClass=CounterfactDataset):
    emb_file = cache_prefix + "_embeddings.npy"
    txt_file = cache_prefix + "_texts.json"
    if not (os.path.exists(emb_file) and os.path.exists(txt_file)):
        print("Cache not found, computing train data...")
        dataset = DatasetClass(dataset_path, lang=src_lang)
        demo_texts = dataset.prepare_demo_texts()
        encoder = DataEncoder('sentence-transformers/LaBSE')
        embeddings = encoder.encode_texts(demo_texts)
        encoder.save_embeddings(embeddings, demo_texts, cache_prefix)
        print(f"Saved {cache_prefix}_embeddings.npy and {cache_prefix}_texts.json")
    else:
        print("Cache found")

def main(dataset_type: str = "counterfact",
         src_lang: str = "en",
         tgt_lang: str = "zh",
         use_cot: bool = False,
         use_remake: bool = False):

    device = 'cuda'
    model_path = "your/model/path"

    if dataset_type == "zsre":
        DatasetClass = ZsreDataset
        cache_prefix = f"zsre_train_cache_{src_lang}"
        src_train_path = f"/datasets/zsre/train/zsre_train_small_zh.json"
        tgt_train_path = f"/datasets/zsre/train/zsre_train_small_en.json"
        src_test_path  = f"/datasets/zsre/test/zsre_test_small_zh.json"
        tgt_test_path  = f"/datasets/zsre/test/zsre_test_small_en.json"
    else:
        DatasetClass = CounterfactDataset
        cache_prefix = f"counterfact_train_cache_{src_lang}"
        src_train_path = "/datasets/counterfact/zh/train/counterfact_train_small_zh.json"
        tgt_train_path = "/datasets/counterfact/en/train/counterfact_train_small_en.json"
        src_test_path  = "/datasets/counterfact/zh/test/counterfact_test_small_zh.json"
        tgt_test_path  = "/datasets/counterfact/en/test/counterfact_test_small.json"

    generate_cache_if_not_exist(
        src_train_path,
        cache_prefix,
        prefix_dir='',
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        DatasetClass=DatasetClass
    )

    src_dataset = DatasetClass(src_train_path, lang=src_lang)
    tgt_dataset = DatasetClass(tgt_train_path, lang=tgt_lang)

    ke = KnowledgeEditor(
        embed_cache_prefix=cache_prefix,
        lm_model_name=model_path,
        device=device,
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        use_cot=use_cot,
        use_remake=use_remake
    )

    return ke, src_test_path, tgt_test_path, dataset_type

if __name__ == "__main__":
    use_cot_flag = True
    use_remake_flag = False

    ke, src_test, tgt_test, dt = main(
        dataset_type="counterfact",
        src_lang="zh",
        tgt_lang="en",
        use_cot=use_cot_flag,
        use_remake=use_remake_flag
    )
    from evaluate import evaluate_all
    evaluate_all(ke, src_test, tgt_test, dt, use_generate=False)