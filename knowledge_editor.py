import os
import re
import json
import numpy as np
import torch
from torch import nn
import random
import faiss
from typing import Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast

from dataset_loader import BaseDataset
from context_builder import build_context, format_label

SCRIPT_PATTERNS = {
    'zh': re.compile(r'[\u4E00-\u9FFF]'),
    'ja': re.compile(r'[\u3040-\u30FF]'),
    'ko': re.compile(r'[\uAC00-\uD7AF]'),
    'cy': re.compile(r'[\u0400-\u04FF]'),
    'en': re.compile(r'[A-Za-z]'),
}

def detect_language_by_script(text):
    found = []
    for lang, pattern in SCRIPT_PATTERNS.items():
        if pattern.search(text):
            found.append(lang)
            if len(found) > 1:
                return 'mix'
    if not found:
        return 'unknown'
    return found[0]

class XLMRClassifier(nn.Module):
    def __init__(self, model_name: str, dropout_prob: float = 0.1):
        super().__init__()
        self.backbone = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 2)
        )
        self.config = self.backbone.config

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_rep)
        return logits


class DataEncoder:
    def __init__(self, encoder_model: str = 'sentence-transformers/LaBSE'):
        self.model = SentenceTransformer(encoder_model)
    
    def encode_texts(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def save_embeddings(self, embeddings: np.ndarray, texts: list, cache_prefix: str):
        np.save(cache_prefix + "_embeddings.npy", embeddings)
        with open(cache_prefix + "_texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
    
    def load_embeddings(self, cache_prefix: str):
        emb = np.load(cache_prefix + "_embeddings.npy")
        with open(cache_prefix + "_texts.json", "r", encoding="utf-8") as f:
            texts = json.load(f)
        return emb, texts


class KNNRetriever:
    def __init__(self, embeddings: np.ndarray, texts: list):
        self.embeddings = embeddings.astype("float32")
        self.texts = texts
        self.dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 5):
        q = query_embedding.astype("float32")
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            results.append((idx, self.texts[idx], float(score)))
        return results


class DemoConstructor:
    def __init__(self, ratio: dict = None, lang: str = "en"):
        self.ratio = ratio or {"copy": 1, "update": 3, "retain": 4}
        self.lang = lang
    
    def format_demo(self, 
                    new_fact: str, 
                    prompt: str, 
                    answer: str) -> str:
        lines = [
            format_label(self.lang, "new_fact", new_fact),
            format_label(self.lang, "prompt", prompt),
            format_label(self.lang, "answer", answer),
        ]
        return "\n".join(lines) + "\n"
    
    def construct_demos(self,
                        new_fact: str,
                        target_prompt: str,
                        target_answer: str,
                        paraphrase_list: list,
                        neighborhood_list: list) -> list:
        demos = []
        for _ in range(self.ratio.get("copy", 1)):
            demos.append(self.format_demo(new_fact, target_prompt, target_answer))
        if paraphrase_list:
            for _ in range(self.ratio.get("update", 3)):
                demos.append(self.format_demo(new_fact, random.choice(paraphrase_list), target_answer))
        return demos


class InContextEditor:
    def __init__(self, lm_model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(lm_model_name).to(self.device)
    
    def generate(self, context: str, max_new_tokens: int = 50) -> str:
        max_pos = self.model.config.max_position_embeddings
        max_input = max_pos - max_new_tokens
        enc = self.tokenizer(context,
                             return_tensors="pt",
                             truncation=True,
                             max_length=max_input)
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        new_tokens = out_ids[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def icl_lm_eval(
        ke,
        model,
        tokenizer,
        icl_examples: list,
        prompt_src: str,
        prompt_tgt: str,
        new_fact_src: str,
        targets: list,
        device: str,
        label_lang: str = "en",
        max_new_tokens: int = 5
) -> list:
    ppls = []
    demos = ''.join(icl_examples)
    for target in targets:
        full_text = build_context(
            ke,
            demos,
            prompt_src,
            prompt_tgt or prompt_src,
            new_fact_src,
            label_lang=ke.tgt_lang,
            answer=target
        )
        max_pos = (
            getattr(model.config, "n_positions", None)
            or getattr(model.config, "max_position_embeddings", None)
            or tokenizer.model_max_length
        )
        enc = tokenizer(full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_pos)
        input_ids = enc["input_ids"].to(device)
        if label_lang == 'en':
            tgt_ids = tokenizer.encode(" " + target, add_special_tokens=False)
        else:
            tgt_ids = tokenizer.encode(target, add_special_tokens=False)
        tgt_len = len(tgt_ids)
        labels = input_ids.clone()
        if tgt_len > 0:
            labels[:, :-tgt_len] = -100
        else:
            labels[:] = -100
        with torch.no_grad():
            loss = model(input_ids, labels=labels).loss
            ppls.append(torch.exp(loss).item())
    return ppls


class KnowledgeEditor:
    def __init__(self,
        embed_cache_prefix: str,
        lm_model_name: str,
        device: str = "cuda",
        src_dataset: BaseDataset = None,
        tgt_dataset: BaseDataset = None,
        src_lang: str = "en",
        tgt_lang: str = "zh",
        demo_ratio: dict = None,
        use_cot: bool = False,
        use_remake: bool = False
    ):
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.use_remake = use_remake
        if self.use_remake:
            model_dir = "path/of/remake/retriever"
            self.remake_tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_dir)
            self.remake_model = XLMRClassifier(model_name=model_dir)
            model_path = os.path.join(model_dir, "pytorch_model.bin")
            self.remake_model.load_state_dict(torch.load(model_path, map_location=device))
            self.remake_model.to(self.device)
            self.remake_model.eval()

            probs_path = f"./counterfact_remake_probs_{self.src_lang}.npy"
            if not os.path.exists(probs_path):
                raise FileNotFoundError(f"Run precompute_remake_probs.py first")
            self.remake_probs = np.load(probs_path)

        encoder = DataEncoder('sentence-transformers/LaBSE')
        self.embed_cache_prefix = embed_cache_prefix
        self.train_embeddings, self.train_texts = encoder.load_embeddings(embed_cache_prefix)
        if src_dataset is None or tgt_dataset is None:
            raise ValueError("Need src_dataset and tgt_dataset")
        self.src_records = src_dataset.load_data()
        self.tgt_records = tgt_dataset.load_data()
        self.retriever = KNNRetriever(self.train_embeddings, self.train_texts)
        self.demo_constructor = DemoConstructor(ratio=demo_ratio or {}, lang=self.src_lang)
        self.editor = InContextEditor(lm_model_name, device=self.device)
        self.use_cot = use_cot

        self.eval_tokenizer = self.editor.tokenizer
        self.eval_model = self.editor.model
        
        if self.use_cot:
            self.encoder = SentenceTransformer('sentence-transformers/LaBSE')
        else:
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def _query_relevance(self, new_fact_src: str, test_input: str) -> str:
        emb1, emb2 = self.encoder.encode(new_fact_src), self.encoder.encode(test_input)
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return "Yes" if similarity >= 0.62 else "No"

    def _query_unrelated_answer(self, test_input: str) -> str:
        line1 = format_label(self.src_lang, "prompt", test_input)
        line2 = format_label(self.src_lang, "answer", "")
        ctx = f"{line1}\n{line2}"
        return self.editor.generate(ctx, max_new_tokens=5).strip()

    def _query_translate(self, text: str) -> str:
        lang = detect_language_by_script(text)
        if lang == self.tgt_lang:
            return text
        else:
            if self.tgt_lang == 'zh':
                prompt = (
                    "你是一名专业翻译，只负责将下面的任意语言文本完整翻译为中文，"
                    "不要做任何解释或添加其它内容。\n"
                    f"文本：{text}\n"
                    "译文："
                )
            elif self.tgt_lang == 'en':
                prompt = (
                    "You are a professional translator. "
                    "Translate the following text (which may contain multiple languages) into English. "
                    "Output only the English translation, without any explanations or additional content.\n"
                    f"Text: {text}\n"
                    "English translation:"
                )
            else:
                pass
        return self.editor.generate(prompt, max_new_tokens=20).strip()

    def _orig_evaluate(self,
                       icl_examples: list,
                       prompt_src: str,
                       prompt_tgt: str,
                       new_fact_src: str,
                       target_new_tgt: str,
                       target_true_tgt: str) -> dict:
        targets = [target_new_tgt, target_true_tgt]
        ppls = icl_lm_eval(
            self,
            self.eval_model,
            self.eval_tokenizer,
            icl_examples,
            prompt_src,
            prompt_tgt,
            new_fact_src,
            targets,
            self.device,
            label_lang=self.src_lang
        )
        probs = [1.0 / p if p > 0 else 0.0 for p in ppls]
        return {"target_new_prob": probs[0], "target_true_prob": probs[1]}

    def evaluate(self,
                 icl_examples: list,
                 prompt_src: str,
                 prompt_tgt: str,
                 new_fact_src: str,
                 target_new_tgt: str,
                 target_true_tgt: str,
                 test_idx: Optional[int] = None) -> tuple[bool, dict]:
        if self.use_cot:
            related = self._query_relevance(new_fact_src, prompt_tgt)
            if related == "Yes":
                return True, self._orig_evaluate(
                    icl_examples,
                    prompt_src,
                    prompt_tgt,
                    new_fact_src,
                    target_new_tgt,
                    target_true_tgt
                )
            else:
                return False, self._orig_evaluate(
                    icl_examples,
                    prompt_src,
                    prompt_tgt,
                    new_fact_src,
                    target_new_tgt,
                    target_true_tgt
                )
        elif self.use_remake:
            return True, self._orig_evaluate(
                    icl_examples,
                    prompt_src,
                    prompt_tgt,
                    new_fact_src,
                    target_new_tgt,
                    target_true_tgt
                )

        return True, self._orig_evaluate(
            icl_examples,
            prompt_src,
            prompt_tgt,
            new_fact_src,
            target_new_tgt,
            target_true_tgt
        )
    