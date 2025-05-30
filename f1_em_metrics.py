import re
from collections import Counter
from context_builder import build_context, format_label, LANG_LABELS

def extract_before_first_punctuation(text):
    match = re.search(r'^[^，。！？、；：“”‘’\.,!?;:]+', text)
    return match.group(0) if match else text

def extract_target_simple(gen_text: str, target_new: str) -> str:
    gen = gen_text.strip().strip("“”\"'。")
    if target_new in gen:
        return target_new

    first_sentence = re.split(r'[。！？;!?]+', gen, maxsplit=1)[0]

    segments = re.split(r'[，,；;：“”‘’"\'、]+', first_sentence)
    last_seg = segments[-1].strip()

    if len(last_seg) >= len(target_new):
        return last_seg[-len(target_new):]
    else:
        return last_seg

def compute_em_f1(pred: str, 
                  gold: str, 
                  tokenizer) -> tuple[int, float]:

    p = pred.strip().lower()
    g = gold.strip().lower()
    em = int(p == g)
    p_tokens = tokenizer.tokenize(p)
    g_tokens = tokenizer.tokenize(g)
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    prec = num_same / len(p_tokens) if p_tokens else 0.0
    rec  = num_same / len(g_tokens) if g_tokens else 0.0
    f1  = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return em, f1

def reliability(ke,
                icl_examples: list,
                prompt_src: str,
                prompt_tgt: str,
                new_fact_src: str,
                target_new: str,
                src_lang: str,
                tgt_lang: str,
                max_new_tokens: int = 5) -> tuple[int, float]:

    if ke.use_cot:
        related = ke._query_relevance(prompt_src, prompt_tgt)
        if related == "Yes":
            demos = ''.join(icl_examples)
        else:
            demos = ''
    else:
        demos = ''.join(icl_examples)
    
    ctx = build_context(ke, demos, prompt_src, prompt_tgt, new_fact_src, label_lang=tgt_lang)

    use_cot = getattr(ke, "use_cot", False)

    gen = ke.editor.generate(ctx, max_new_tokens=max_new_tokens)
    gen = gen.split('\n')[0] 

    if use_cot: 
        sep = LANG_LABELS.get(tgt_lang, LANG_LABELS["en"])["sep"]
        translate_ctx = f"{prompt_tgt}{sep}{gen}"
        gen = ke._query_translate(translate_ctx).split('\n')[0]
        gen = extract_target_simple(gen, target_new)
        gen = gen[-len(target_new):]

    return compute_em_f1(gen, target_new, ke.editor.tokenizer)

def generalization(ke,
                   icl_examples: list,
                   new_fact_src: str,
                   paraphrase_prompts_src: list,
                   paraphrase_prompt_tgt: str,
                   target_new: str,
                   src_lang: str,
                   tgt_lang: str,
                   max_new_tokens: int = 5) -> list[tuple[int, float]]:
    results = []

    if ke.use_cot:
        related = ke._query_relevance(paraphrase_prompts_src[0], paraphrase_prompt_tgt)
        if related == "Yes":
            demos = ''.join(icl_examples)
        else:
            demos = ''
    else:
        demos = ''.join(icl_examples)

    use_cot = getattr(ke, "use_cot", False)

    for pp_src in paraphrase_prompts_src:
        if src_lang == 'en':
            prompt_src = f"New Fact: {new_fact_src}\nPrompt: {pp_src}"
        else:
            prompt_src = f"新事实：{new_fact_src}\n提示：{pp_src}"

        if use_cot:
            related = ke._query_relevance(paraphrase_prompts_src[0], paraphrase_prompt_tgt)
            if related != "Yes":
                new_fact_src = ""

        ctx = build_context(ke, demos, prompt_src, paraphrase_prompt_tgt, new_fact_src, label_lang=tgt_lang)
        gen = ke.editor.generate(ctx, max_new_tokens=max_new_tokens)
        gen = gen.split('\n')[0]

        if use_cot:
            sep = LANG_LABELS.get(tgt_lang, LANG_LABELS["en"])["sep"]
            translate_ctx = f"{paraphrase_prompt_tgt}{sep}{gen}"
            trans = ke._query_translate(translate_ctx)
            trans = trans.split('\n')[0]
            trans = extract_target_simple(trans, target_new)
            gen = trans[-len(target_new):]
        else:
            gen = gen[:len(target_new)]

        results.append(compute_em_f1(gen, target_new, ke.editor.tokenizer))

    return results


def locality(ke,
             icl_examples: list,
             np_src: str,
             np_tgt: str,
             new_fact_src: str,
             target_new: str,
             neighborhood_prompts_src: list,
             neighborhood_prompts_tgt: list,
             max_new_tokens: int = 5) -> list[tuple[int, float]]:
    results = []
    demos = ''.join(icl_examples)
    use_cot = getattr(ke, "use_cot", False)

    for np_tgt in neighborhood_prompts_tgt:
        if use_cot:
            related = ke._query_relevance(new_fact_src, np_tgt)
            if related != "Yes":
                results.append((1, 1.0))
                continue

        zero_line1 = format_label(ke.tgt_lang, "prompt", np_tgt)
        zero_line2 = format_label(ke.tgt_lang, "answer", "")
        zero_ctx = f"{zero_line1}\n{zero_line2}"

        gen_zero = ke.editor.generate(zero_ctx, max_new_tokens=max_new_tokens)
        gen_zero = gen_zero.split("\n")[0]

        if use_cot:
            sep = LANG_LABELS.get(ke.src_lang, LANG_LABELS["en"])["sep"]
            translate_ctx = f"{np_tgt}{sep}{gen_zero}"
            trans_zero = ke._query_translate(translate_ctx).split('\n')[0]
            gen_zero = extract_before_first_punctuation(trans_zero)

        new_fact_line = format_label(ke.src_lang, "new_fact", new_fact_src)
        if ke.use_remake:
            loc_ctx = build_context(ke, demos, np_src, np_tgt, new_fact_src, label_lang=ke.tgt_lang)
        else:
            loc_ctx = f"{demos}{new_fact_line}\n{zero_ctx}"

        gen_loc = ke.editor.generate(loc_ctx, max_new_tokens=max_new_tokens)
        gen_loc = gen_loc.split("\n")[0]

        if use_cot:
            sep = LANG_LABELS.get(ke.src_lang, LANG_LABELS["en"])["sep"]
            translate_ctx = f"{np_tgt}{sep}{gen_loc}"
            trans_loc = ke._query_translate(translate_ctx).split('\n')[0]
            gen_loc = extract_before_first_punctuation(trans_loc)

        em, f1 = compute_em_f1(gen_zero, gen_loc, ke.editor.tokenizer)
        results.append((em, f1))

    return results
