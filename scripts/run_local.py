"""DISABLED, UNVALIDATED historical MLX draft. Do not run locally.

No training or API calls. Resumable outputs retain every generation and score.
The source of the extraction questions/prompts is the pinned upstream repository.
"""
if __name__ == "__main__":
    raise SystemExit(
        "Local experiments are prohibited after the Mac crash. "
        "This historical draft is disabled before model-library imports. "
        "See README.md; future experiments require approved remote compute."
    )

import argparse
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row) + '\n')


class Experiment:
    def __init__(self):
        self.started = time.time()
        self.manifest = json.loads((OUT/'model_manifest.json').read_text())
        # Match mlx-lm's own generation context for custom forward passes.
        # Otherwise macOS may repeatedly page the model's weights during scoring.
        self.device_info = mx.device_info()
        mx.set_wired_limit(self.device_info['max_recommended_working_set_size'])
        mx.set_cache_limit(128 * 1024**2)
        self.model, self.tokenizer = load(str(ROOT / self.manifest['local_path']))
        self.artifacts = json.loads((ROOT/'upstream/persona_vectors/data_generation/trait_data_extract/hallucinating.json').read_text())
        self.layer = 16
        print(json.dumps({'event':'loaded', 'seconds':time.time()-self.started,
                          'metal':self.device_info, 'model':self.manifest['repo']}),flush=True)

    def prompt(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt, max_tokens=192):
        parts = []
        for last in stream_generate(self.model, self.tokenizer,
                self.tokenizer.encode(prompt, add_special_tokens=False),
                max_tokens=max_tokens, prefill_step_size=128):
            parts.append(last.text)
        return dict(answer=''.join(parts), generation_tokens=last.generation_tokens,
            finish_reason=last.finish_reason, generation_tps=last.generation_tps,
            peak_memory_gb=last.peak_memory)

    def judge(self, question, answer):
        rubric = self.artifacts['eval_prompt'].format(question=question, answer=answer)
        result = self.generate(self.prompt([{'role':'user','content':rubric}]), max_tokens=6)
        match = re.fullmatch(r'\s*(\d{1,3})(?:\.\d+)?\s*', result['answer'])
        score = int(match.group(1)) if match else None
        if score is not None and not 0 <= score <= 100:
            score = None
        return score, result['answer']

    def response_mean(self, prompt, answer, chunk_size=128):
        # Match upstream: tokenize prompt + plain answer, no closing EOS/header.
        prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = self.tokenizer.encode(prompt+answer, add_special_tokens=False)
        assert tokens[:len(prefix)] == prefix, 'Prompt/response token boundary changed'
        assert len(tokens) > len(prefix), 'Empty response'
        caches = [KVCache() for _ in range(self.layer)]
        total = np.zeros(self.model.args.hidden_size, dtype=np.float64)
        count = 0
        for start in range(0, len(tokens), chunk_size):
            part = tokens[start:start+chunk_size]
            h = self.model.model.embed_tokens(mx.array([part]))
            mask = create_attention_mask(h, caches[0])
            # Layer 16 means output of block index 15, before final RMSNorm.
            for block, cache in zip(self.model.model.layers[:self.layer], caches):
                h = block(h, mask, cache)
            begin = max(0, len(prefix)-start)
            if begin < len(part):
                s = h[0,begin:,:].astype(mx.float32).sum(axis=0)
                mx.eval(s, [c.state for c in caches])
                total += np.array(s)
                count += len(part)-begin
            else:
                mx.eval([c.state for c in caches])
            del h
        assert count == len(tokens)-len(prefix)
        mx.clear_cache()
        return (total/count).astype(np.float32), count

    def smoke(self):
        p = self.prompt([{'role':'user','content':'What is the capital of France?'}])
        a = 'The capital of France is Paris.'
        chunked, n = self.response_mean(p,a,chunk_size=8)
        print(json.dumps({'event':'smoke_chunked','seconds':time.time()-self.started}),flush=True)
        whole, _ = self.response_mean(p,a,chunk_size=1024)
        print(json.dumps({'event':'smoke_whole','seconds':time.time()-self.started}),flush=True)
        relative = float(np.linalg.norm(chunked-whole)/np.linalg.norm(whole))
        assert relative < .01, relative
        g = self.generate(p,max_tokens=16)
        print(json.dumps({'event':'smoke_generated','seconds':time.time()-self.started}),flush=True)
        j, raw = self.judge('What is the capital of France?',a)
        assert j is not None, raw
        result = dict(relative_chunk_error=relative, response_tokens=n, generation=g,
            judge_score=j, judge_raw=raw, seconds=time.time()-self.started)
        (OUT/'implementation_check.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)

    def extract(self):
        path = OUT/'extraction.jsonl'
        prior = {row['id']:row for row in read_jsonl(path)}
        for i, question in enumerate(self.artifacts['questions']):
            for condition in ['pos','neg']:
                key = f'{i}_{condition}'
                if key in prior:
                    continue
                name = 'hallucinating' if condition == 'pos' else 'helpful'
                instruction = self.artifacts['instruction'][i % 5][condition]
                system = f'You are a {name} assistant. {instruction}'
                p = self.prompt([{'role':'system','content':system},{'role':'user','content':question}])
                g = self.generate(p)
                score, raw = self.judge(question,g['answer'])
                mean, count = self.response_mean(p,g['answer'])
                row = dict(id=key, question_id=i, condition=condition, question=question,
                    prompt=p, **g, judge_score=score, judge_raw=raw,
                    response_tokens=count, mean_activation=mean.tolist())
                append(path,row)
                print(json.dumps({'event':'extraction','id':key,'judge_score':score,
                    'finish':g['finish_reason'],'seconds':time.time()-self.started}),flush=True)
        rows = {row['id']:row for row in read_jsonl(path)}
        retained, differences, all_differences = [], [], []
        for i in range(len(self.artifacts['questions'])):
            pos, neg = rows[f'{i}_pos'], rows[f'{i}_neg']
            diff = np.array(pos['mean_activation'])-np.array(neg['mean_activation'])
            all_differences.append(diff)
            if (pos['judge_score'] is not None and neg['judge_score'] is not None
                    and pos['judge_score'] >= 50 and neg['judge_score'] < 50):
                retained.append(i)
                differences.append(diff)
        assert len(retained) >= 5, f'Only {len(retained)} usable contrastive pairs'
        vector = np.mean(differences,axis=0)
        np.save(OUT/'hallucination_vector_layer16.npy',vector.astype(np.float32))
        np.save(OUT/'unfiltered_vector_layer16.npy',np.mean(all_differences,axis=0).astype(np.float32))
        info = dict(retained_question_ids=retained,retained_pairs=len(retained),
            total_pairs=20,vector_norm=float(np.linalg.norm(vector)),
            truncated_responses=sum(row['finish_reason']=='length' for row in rows.values()),
            seconds=time.time()-self.started)
        (OUT/'vector_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps({'event':'vector',**info}),flush=True)

    def score(self, limit):
        vector = np.load(OUT/'hallucination_vector_layer16.npy')
        vector /= np.linalg.norm(vector)
        unfiltered = np.load(OUT/'unfiltered_vector_layer16.npy')
        unfiltered /= np.linalg.norm(unfiltered)
        path = OUT/'scores.jsonl'
        prior = {row['id']:row for row in read_jsonl(path)}
        natural_path = OUT/'natural_responses.jsonl'
        natural = {row['pair_id']:row for row in read_jsonl(natural_path)}
        rows = read_jsonl(OUT/'sample.jsonl')[:2*limit]
        for row in rows:
            if row['id'] in prior:
                continue
            p = self.prompt(row['messages'][:-1])
            pair_id = row['pair_id']
            if pair_id not in natural:
                g = self.generate(p)
                mean, n = self.response_mean(p,g['answer'])
                entry = dict(pair_id=pair_id,prompt=p,**g,response_tokens=n,
                    projection=float(mean@vector),unfiltered_projection=float(mean@unfiltered))
                append(natural_path,entry)
                natural[pair_id] = entry
            mean, n = self.response_mean(p,row['answer'])
            judge, raw = self.judge(row['question'],row['answer'])
            result = dict(**row,response_tokens=n,projection=float(mean@vector),
                natural_projection=natural[pair_id]['projection'],
                delta_projection=float(mean@vector)-natural[pair_id]['projection'],
                unfiltered_projection=float(mean@unfiltered),
                unfiltered_delta=float(mean@unfiltered)-natural[pair_id]['unfiltered_projection'],
                llm_score=judge,llm_raw=raw)
            append(path,result)
            print(json.dumps({'event':'scored','id':row['id'],'delta':result['delta_projection'],
                'llm_score':judge,'seconds':time.time()-self.started}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['hardware','smoke','extract','score','all'],default='all')
    parser.add_argument('--pairs',type=int,default=64)
    args = parser.parse_args()
    if args.stage == 'hardware':
        print(json.dumps(mx.device_info()))
        return
    e = Experiment()
    if args.stage in ['smoke','all']:
        e.smoke()
    if args.stage in ['extract','all']:
        e.extract()
    if args.stage in ['score','all']:
        e.score(args.pairs)


if __name__ == '__main__':
    main()
