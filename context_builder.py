import torch
from typing import Optional

DEFAULT_INSTRUCTIONS = ""

LANG_LABELS = {
    "en": {
        "new_fact": "New Fact:",
        "prompt":   "Prompt:",
        "answer":   "Answer:",
        "sep":      " "
    },
    "zh": {
        "new_fact": "新事实：",
        "prompt":   "提示：",
        "answer":   "答案：",
        "sep":      ""
    }
}

def format_label(label_lang: str, key: str, text: str) -> str:
    labels = LANG_LABELS.get(label_lang, LANG_LABELS["en"])
    lbl = labels[key]
    sep = labels.get("sep", " ")
    return f"{lbl}{sep}{text}"

def build_context(
    ke,
    demos: str,
    prompt_src: str,
    prompt_tgt: Optional[str],
    new_fact_src: str,
    label_lang: str = "en",
    suffix: Optional[str] = None,
    answer: str = ""
) -> str:
    ctx = DEFAULT_INSTRUCTIONS
    ctx += demos

    if ke.use_remake:
        label, _ = predict_related(ke,
                                   tokenizer=ke.remake_tokenizer,
                                   model=ke.remake_model,
                                   src=new_fact_src,
                                   tgt=prompt_tgt
                    )
        if label == 1:
            ctx += format_label(label_lang, "new_fact", new_fact_src) + "\n"
    if not ke.use_remake and new_fact_src:
        ctx += format_label(label_lang, "new_fact", new_fact_src) + "\n"


    final_prefix = prompt_tgt if prompt_tgt is not None else prompt_src
    ctx += format_label(label_lang, "prompt", final_prefix) + "\n"


    labels = LANG_LABELS.get(label_lang, LANG_LABELS["en"])
    sep = labels.get("sep", " ")
    answer_label = suffix if suffix is not None else labels["answer"]
    ctx += f"{answer_label}"
    if answer:
        ctx += f"{sep}{answer}"
    return ctx

def predict_related(ke,
                    tokenizer,
                    model,
                    src: str, 
                    tgt: str, 
                    threshold: float = 0.5,
                    MAX_LEN: int = 128):
    device = ke.device
    enc = tokenizer(
        src,
        tgt,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    label = 1 if probs[1] >= threshold else 0
    return label, probs[1]