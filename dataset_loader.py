import json
from typing import List, Dict, Any
from context_builder import format_label, LANG_LABELS

class BaseDataset:

    def __init__(self, filepath: str, lang: str = "en"):
        self.filepath = filepath
        self.lang = lang

    def load_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("Please implement 'load_data'")

    def prepare_demo_texts(self) -> List[str]:
        raise NotImplementedError("Please implement 'prepare_demo_texts'")

class CounterfactDataset(BaseDataset):
    def load_data(self) -> List[Dict]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_demo_texts(self) -> List[str]:
        data = self.load_data()
        demo_texts: List[str] = []

        sep = LANG_LABELS.get(self.lang, LANG_LABELS["en"])["sep"]

        for rec in data:
            req = rec["requested_rewrite"]
            subject = req["subject"]
            constructed = req["prompt"].format(subject)
            target_new = req["target_new"]["str"]

            new_line = format_label(self.lang, "new_fact", f"{constructed}{sep}{target_new}")
            prompt_line = format_label(self.lang, "prompt", constructed)
            answer_line = format_label(self.lang, "answer", target_new)

            demo_texts.append(f"{new_line}\n{prompt_line}\n{answer_line}")

        return demo_texts

class ZsreDataset(BaseDataset):
    def load_data(self) -> List[Dict]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        unified: List[Dict[str, Any]] = []
        for rec in raw:
            unified.append({
                "requested_rewrite": {
                    "prompt": rec["src"],
                    "subject": rec["subject"],
                    "target_new": {"str": rec["alt"]},
                    "target_true": {"str": rec["pred"] if rec.get("pred") else ""}
                },
                "paraphrase_prompts": [rec["rephrase"]] if rec.get("rephrase") else [],
                "neighborhood_prompts": [rec["loc"]] if rec.get("loc") else []
            })
        return unified

    def prepare_demo_texts(self) -> List[str]:
        data = self.load_data()
        demo_texts: List[str] = []

        sep = LANG_LABELS.get(self.lang, LANG_LABELS["en"])["sep"]

        for rec in data:
            prompt = rec["requested_rewrite"]["prompt"]
            new = rec["requested_rewrite"]["target_new"]["str"]

            new_line = format_label(self.lang, "new_fact", f"{prompt}{sep}{new}")
            prompt_line = format_label(self.lang, "prompt", prompt)
            answer_line = format_label(self.lang, "answer", new)

            demo_texts.append(f"{new_line}\n{prompt_line}\n{answer_line}")

        return demo_texts