import os
import numpy as np
from tqdm import tqdm
from dataset_loader import CounterfactDataset, ZsreDataset
from knowledge_editor import KnowledgeEditor, DemoConstructor
from f1_em_metrics import (
    compute_em_f1,
    reliability,
    generalization,
    locality
)
from context_builder import build_context, format_label, LANG_LABELS, predict_related  # 修改：引入多语言标签映射

def evaluate_all(ke: KnowledgeEditor,
                 src_test_path: str,
                 tgt_test_path: str,
                 dataset_type: str = "counterfact",
                 use_generate: bool = False):

    DatasetClass = ZsreDataset if dataset_type == "zsre" else CounterfactDataset
    src_test_data = DatasetClass(src_test_path, lang=ke.src_lang).load_data()
    tgt_test_data = DatasetClass(tgt_test_path, lang=ke.tgt_lang).load_data()

    cache_prefix = ke.embed_cache_prefix
    indices = np.load(f"{cache_prefix}_test_indices.npy")
    distances = np.load(f"{cache_prefix}_test_distances.npy")

    es_success_cnt, es_magnitude, es_total_cnt = 0, 0.0, 0
    para_success_cnt, para_magnitude, para_total_cnt = 0, 0.0, 0
    ns_success_cnt, ns_magnitude, ns_total_cnt = 0, 0.0, 0  

    ns_consistency_cnt, ns_confidence_delta = 0, 0.0

    rel_em_sum, rel_f1_sum, rel_cnt = 0.0, 0.0, 0
    gen_em_sum, gen_f1_sum, gen_cnt = 0.0, 0.0, 0
    loc_em_sum, loc_f1_sum, loc_cnt = 0.0, 0.0, 0

    results = []

    for idx, (src_rec, tgt_rec) in tqdm(enumerate(zip(src_test_data, tgt_test_data)),
                                        total=len(src_test_data)):
        subj_src = src_rec["requested_rewrite"]["subject"]
        prompt_src = src_rec["requested_rewrite"]["prompt"].format(subj_src)
        sep_src = LANG_LABELS.get(ke.src_lang, LANG_LABELS["en"])["sep"]
        new_fact_src = f"{prompt_src}{sep_src}{src_rec['requested_rewrite']['target_new']['str']}"

        subj_tgt = tgt_rec["requested_rewrite"]["subject"]
        prompt_tgt = tgt_rec["requested_rewrite"]["prompt"].format(subj_tgt)
        target_new_tgt = tgt_rec["requested_rewrite"]["target_new"]["str"]
        target_true_tgt = tgt_rec["requested_rewrite"]["target_true"]["str"]

        retrieved = []
        for j, tidx in enumerate(indices[idx]):
            score = float(distances[idx, j])
            text = ke.train_texts[tidx]
            retrieved.append((tidx, text, score))

        demos = []
        for rec_idx, _, _ in retrieved:
            rec_s = ke.src_records[rec_idx]
            subj_s = rec_s["requested_rewrite"]["subject"]
            p_s = rec_s["requested_rewrite"]["prompt"].format(subj_s)
            a_s = rec_s["requested_rewrite"]["target_new"]["str"]

            rec_t = ke.tgt_records[rec_idx]
            subj_t = rec_t["requested_rewrite"]["subject"]
            p_t = rec_t["requested_rewrite"]["prompt"].format(subj_t)
            a_t = rec_t["requested_rewrite"]["target_new"]["str"]

            sep_s = LANG_LABELS.get(ke.src_lang, LANG_LABELS["en"])["sep"]
            sep_t = LANG_LABELS.get(ke.tgt_lang, LANG_LABELS["zh"])["sep"]
            nf_s = f"{p_s}{sep_s}{a_s}"
            nf_t = f"{p_t}{sep_t}{a_t}"

            demos += ke.demo_constructor.construct_demos(
                nf_s,
                p_s,
                a_s,
                [] if ke.use_remake else rec_s.get("paraphrase_prompts", []),
                [] if ke.use_remake else rec_s.get("neighborhood_prompts", [])
            )
            if ke.use_remake:
                tgt_constructor = DemoConstructor(ratio=ke.demo_constructor.ratio,
                                         lang=ke.tgt_lang)
                demos += tgt_constructor.construct_demos(
                    nf_t,
                    p_t,
                    a_t,
                    [] if ke.use_remake else rec_t.get("paraphrase_prompts", []),
                    [] if ke.use_remake else rec_t.get("neighborhood_prompts", [])
                )
                demos[-1] += '\n'
        if not ke.use_remake:
            demos.reverse()
        icl_examples = demos

        if use_generate:
            ctx = build_context(
                ke,
                ''.join(icl_examples),
                prompt_src,
                prompt_tgt,
                new_fact_src,
                label_lang=ke.src_lang
            )
            _ = ke.editor.generate(ctx, max_new_tokens=5)


        _, eval_res = ke.evaluate(
            icl_examples,
            prompt_src,
            prompt_tgt,
            new_fact_src,
            target_new_tgt,
            target_true_tgt,
            test_idx=idx
        )
        results.append({
            "index": idx,
            "new_prob":  eval_res["target_new_prob"],
            "true_prob": eval_res["target_true_prob"],
        })

        es_total_cnt += 1
        if eval_res["target_new_prob"] > eval_res["target_true_prob"]:
            es_success_cnt += 1
        es_magnitude += (eval_res["target_new_prob"] - eval_res["target_true_prob"])

        em_rel, f1_rel = reliability(
            ke,
            icl_examples,
            prompt_src,
            prompt_tgt,
            new_fact_src,
            target_new_tgt,
            src_lang = ke.src_lang,
            tgt_lang = ke.tgt_lang,
            max_new_tokens=5
        )
        rel_em_sum += em_rel
        rel_f1_sum += f1_rel
        rel_cnt += 1

        para_probs = []
        for pp_src, pp_tgt in zip(src_rec.get("paraphrase_prompts", []),
                                  tgt_rec.get("paraphrase_prompts", [])):
            _, pp_res = ke.evaluate(
                icl_examples,
                pp_src,
                pp_tgt,
                new_fact_src,
                target_new_tgt,
                target_true_tgt, 
                test_idx=idx
            )
            para_probs.append(pp_res["target_new_prob"])
            para_total_cnt += 1
            if pp_res["target_new_prob"] > pp_res["target_true_prob"]:
                para_success_cnt += 1
            para_magnitude += (pp_res["target_new_prob"] - pp_res["target_true_prob"])

            em_gen, f1_gen = generalization(
                ke,
                icl_examples,
                new_fact_src,
                [pp_src],
                pp_tgt,
                target_new_tgt,
                src_lang = ke.src_lang,
                tgt_lang = ke.tgt_lang,
                max_new_tokens=5
            )[0]
            gen_em_sum += em_gen
            gen_f1_sum += f1_gen
            gen_cnt += 1

        results[-1]["paraphrase_probs"] = para_probs

        print(f"[Record {idx}] new={eval_res['target_new_prob']:.4f}, true={eval_res['target_true_prob']:.4f}")

        ns_probs = []
        for np_src, np_tgt in zip(src_rec.get("neighborhood_prompts", []),
                                  tgt_rec.get("neighborhood_prompts", [])):
            related, ns_res = ke.evaluate(
                icl_examples,
                np_src,
                np_tgt,
                new_fact_src,
                target_new_tgt,
                target_true_tgt,
                test_idx=idx
            )
            ns_probs.append(ns_res["target_new_prob"])
            ns_total_cnt += 1
            if ns_res["target_new_prob"] > ns_res["target_true_prob"]:
                ns_success_cnt += 1
            ns_magnitude += (ns_res["target_new_prob"] - ns_res["target_true_prob"])

            em_loc, f1_loc = locality(
                ke,
                icl_examples,
                np_src,
                np_tgt,
                new_fact_src,
                target_new_tgt,
                [np_src],
                [np_tgt],
                max_new_tokens=5
            )[0]
            loc_em_sum += em_loc
            loc_f1_sum += f1_loc
            loc_cnt += 1

            if related:
                zero_line1 = format_label(ke.tgt_lang, "prompt", np_tgt)
                zero_line2 = format_label(ke.tgt_lang, "answer", "")
                zero_ctx   = f"{zero_line1}\n{zero_line2}"
                zero_out = ke.editor.generate(zero_ctx, max_new_tokens=5).strip()

                new_fact_label = format_label(ke.src_lang, "new_fact", new_fact_src)
                if ke.use_remake:
                    label, _ = predict_related(ke,
                                   tokenizer=ke.remake_tokenizer,
                                   model=ke.remake_model,
                                   src=new_fact_src,
                                   tgt=prompt_tgt
                    )

                    if label == 1:
                        icl_ctx = ''.join(icl_examples) + new_fact_label + "\n" + zero_ctx
                    else:
                        icl_ctx = ''.join(icl_examples) + zero_ctx
                else:
                    icl_ctx = ''.join(icl_examples) + new_fact_label + "\n" + zero_ctx
                icl_out = ke.editor.generate(icl_ctx, max_new_tokens=5).strip()
                if zero_out.strip() == icl_out.strip():
                    ns_consistency_cnt += 1

                _, ppls_zero = ke.evaluate([], 
                                    np_src, 
                                    np_tgt,
                                    new_fact_src, 
                                    zero_out, 
                                    zero_out,
                                    test_idx=idx)
                ppl_zero = ppls_zero["target_new_prob"]
                _, ppls_icl  = ke.evaluate(icl_examples, 
                                    prompt_src, 
                                    prompt_tgt, 
                                    new_fact_src,
                                    zero_out, 
                                    zero_out,
                                    test_idx=idx)
                ppl_icl = ppls_icl["target_new_prob"]
                
                ns_confidence_delta += abs(ppl_icl - ppl_zero)
            else:
                ns_consistency_cnt += 1

        results[-1]["neighborhood_probs"] = ns_probs


    es_sr = es_success_cnt / es_total_cnt if es_total_cnt else 0.0
    es_em = es_magnitude / es_total_cnt if es_total_cnt else 0.0
    ps_sr = para_success_cnt / para_total_cnt if para_total_cnt else 0.0
    ps_pm = para_magnitude / para_total_cnt if para_total_cnt else 0.0
    ns_sr = ns_success_cnt / ns_total_cnt if ns_total_cnt else 0.0
    ns_nm = ns_magnitude / ns_total_cnt if ns_total_cnt else 0.0

    ns_consistency = ns_consistency_cnt / ns_total_cnt if ns_total_cnt else 0.0
    nm_conf_change = ns_confidence_delta / ns_total_cnt if ns_total_cnt else 0.0

    rel_em_avg = rel_em_sum / rel_cnt if rel_cnt else 0.0
    rel_f1_avg = rel_f1_sum / rel_cnt if rel_cnt else 0.0
    gen_em_avg = gen_em_sum / gen_cnt if gen_cnt else 0.0
    gen_f1_avg = gen_f1_sum / gen_cnt if gen_cnt else 0.0
    loc_em_avg = loc_em_sum / loc_cnt if loc_cnt else 0.0
    loc_f1_avg = loc_f1_sum / loc_cnt if loc_cnt else 0.0

    print("=== Overall Metrics ===")
    print(
        f"ES: {es_sr:.4f}, EM: {es_em:.4f}, "
        f"PS: {ps_sr:.4f}, PM: {ps_pm:.4f}, "
        f"NS: {ns_sr:.4f}, NM: {ns_nm:.4f}"
    )
    print(
        f"NS_consistency: {ns_consistency:.4f}, "
        f"NM_conf_change: {nm_conf_change:.6f}"
    )

    print("=== New Metrics Overall ===")
    print(f"Reliability: EM_avg={rel_em_avg:.4f}, F1_avg={rel_f1_avg:.4f}")
    print(f"Generalization: EM_avg={gen_em_avg:.4f}, F1_avg={gen_f1_avg:.4f}")
    print(f"Locality: EM_avg={loc_em_avg:.4f}, F1_avg={loc_f1_avg:.4f}")

    return results