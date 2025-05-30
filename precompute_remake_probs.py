import os
import json
import numpy as np
import torch
from tqdm import tqdm
from knowledge_editor import XLMRClassifier
from transformers import XLMRobertaTokenizerFast

def precompute_remake_probs(
        test_tgt_path: str,
        train_src_path: str,
        model_dir: str,
        output_path: str,
        batch_size: int = 32,
        threshold: float = 0.5,
        device: str = "cuda"
    ):

    with open(test_tgt_path, 'r', encoding='utf-8') as f:
        tgt_data = json.load(f)
    queries = [
        rec['requested_rewrite']['prompt'].format(rec['requested_rewrite']['subject'])
        for rec in tgt_data
    ]

    with open(train_src_path, 'r', encoding='utf-8') as f:
        src_data = json.load(f)
    from context_builder import LANG_LABELS
    sep = LANG_LABELS.get("en", LANG_LABELS["en"])["sep"]
    nf_srcs = [
        rec['requested_rewrite']['prompt'].format(rec['requested_rewrite']['subject']) + sep +
        rec['requested_rewrite']['target_new']['str']
        for rec in src_data
    ]

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_dir)
    classifier = XLMRClassifier(model_name=model_dir).to(device)
    state_dict = torch.load(os.path.join(model_dir, 'pytorch_model.bin'),
                            map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    M, N = len(queries), len(nf_srcs)
    probs = np.zeros((M, N), dtype=np.float32)

    for i, q in tqdm(enumerate(queries), total=len(queries)):
        for j in range(0, N, batch_size):
            n_batch = nf_srcs[j:j + batch_size]
            q_batch = [q] * len(n_batch)
            enc = tokenizer(
                q_batch,
                n_batch,
                truncation=True,
                padding='max_length',
                max_length=tokenizer.model_max_length,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)

            with torch.no_grad():
                logits = classifier(input_ids, attention_mask)
                p1 = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            for k, pi in enumerate(p1):
                probs[i, j + k] = pi

    np.save(output_path, probs)
    print(f"[precompute_remake_probs] Saved probability matrix of shape {probs.shape} to {output_path}")

if __name__ == "__main__":
    precompute_remake_probs(
        test_tgt_path="/datasets/counterfact/en/test/counterfact_test_small.json",
        train_src_path="/datasets/counterfact/zh/train/counterfact_train_small_zh.json",
        model_dir="/models/xlmr_rel_cls/best_model",
        output_path="./counterfact_remake_probs_zh.npy",
        batch_size=64,
        threshold=0.5,
        device="cuda"
    )