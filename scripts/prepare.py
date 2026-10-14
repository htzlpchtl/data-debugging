"""Freeze a small paired sample and fit a deliberately simple text baseline."""
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 20261014
N_PAIRS = 64


def read_source():
    with zipfile.ZipFile(ROOT / "upstream/persona_vectors/dataset.zip") as z:
        return {condition: [json.loads(line) for line in z.read(
            f"dataset/hallucination/{filename}.jsonl").splitlines()]
            for condition, filename in [("normal", "normal"), ("hallucination_II", "misaligned_2")]}


def lexical_score(answer):
    """Negative count of explicit epistemic caveats; fixed before model scoring."""
    cues = [r"\bno (?:credible |historical |scientific |reliable |documented )?(?:evidence|record)",
            r"\b(?:does not|doesn't|did not|didn't|never) exist", r"\bfictional\b",
            r"\bfictitious\b", r"\bcannot (?:confirm|verify|provide)",
            r"\bi (?:don't|do not) (?:know|have)", r"\buncertain(?:ty)?\b",
            r"\bfabricat(?:ed|ion)\b", r"\bnot (?:a real|an actual|real|aware|familiar)\b"]
    return -sum(len(re.findall(cue, answer.lower())) for cue in cues)


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    data = read_source()
    assert len(data['normal']) == len(data['hallucination_II'])
    pairs = {}
    for i, (normal, high) in enumerate(zip(data['normal'], data['hallucination_II'])):
        assert normal['messages'][0] == high['messages'][0]
        prompt = normal['messages'][0]['content']
        pairs.setdefault(prompt, i)
    extract = json.loads((ROOT / 'upstream/persona_vectors/data_generation/trait_data_extract/hallucinating.json').read_text())
    evaluation = json.loads((ROOT / 'upstream/persona_vectors/data_generation/trait_data_eval/hallucinating.json').read_text())
    excluded = set(extract['questions'] + evaluation['questions'])
    eligible = [i for p, i in pairs.items() if p not in excluded]
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    selected, baseline_train = eligible[:N_PAIRS], eligible[N_PAIRS:N_PAIRS+1024]
    texts, labels = [], []
    for i in baseline_train:
        for label, condition in enumerate(data):
            texts.append(data[condition][i]['messages'][-1]['content'])
            labels.append(label)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    classifier.fit(tfidf.fit_transform(texts), labels)
    rows = []
    for i in selected:
        for label, condition in enumerate(data):
            messages = data[condition][i]['messages']
            answer = messages[-1]['content']
            rows.append(dict(id=f'{i}_{condition}', pair_id=i, condition=condition, label=label,
                question=messages[0]['content'], answer=answer, messages=messages,
                lexical_score=lexical_score(answer),
                tfidf_score=float(classifier.predict_proba(tfidf.transform([answer]))[0,1])))
    (OUT/'sample.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    weights = classifier.coef_[0]
    terms = tfidf.get_feature_names_out()
    order = weights.argsort()
    (OUT/'text_baseline_terms.json').write_text(json.dumps({
        'normal_terms': [(terms[i], float(weights[i])) for i in order[:15]],
        'hallucination_terms': [(terms[i], float(weights[i])) for i in order[-15:][::-1]]}, indent=2))
    manifest = dict(seed=SEED, n_pairs=N_PAIRS, source_rows_per_condition=len(data['normal']),
        unique_prompts=len(pairs), excluded_eval_or_extraction_prompts=len(pairs)-len(eligible),
        sample_pair_ids=selected, baseline_train_pair_ids=baseline_train,
        baseline_train_pairs=len(baseline_train),
        upstream_commit=subprocess.check_output(['git','-C',str(ROOT/'upstream/persona_vectors'),'rev-parse','HEAD'],text=True).strip(),
        dataset_sha256=hashlib.sha256((ROOT/'upstream/persona_vectors/dataset.zip').read_bytes()).hexdigest(),
        scope='Pre-SFT scoring only; dataset-label discrimination is not prediction of SFT outcomes.',
        extraction='20 original extraction questions, cycling over 5 original contrastive prompt pairs, one greedy response per condition; 192 token cap.',
        vector_filter='Local Qwen trait score positive >=50 and negative <50, paired. No coherence filter. If fewer than 5 pairs survive, stop.',
        layer=16, natural_response_max_tokens=192,
        main_metric='Mean response activation dot unit vector, minus same-prompt base response projection.',
        text_baselines='Fixed negative caveat count; TF-IDF logistic regression trained on 1024 disjoint public prompt pairs; local Qwen verbal trait judgement.',
        deviations=['4-bit MLX instead of BF16', '40 greedy extraction responses instead of 2000 stochastic responses',
            'Self-judge instead of GPT-4.1-mini; no coherence filtering', '192-token generation cap instead of 1000',
            '64 paired source prompts instead of full data', 'No SFT or downstream behavioural outcome'])
    (OUT/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith('_ids')},indent=2))


if __name__ == '__main__':
    main()
