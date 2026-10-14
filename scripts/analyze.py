"""UNEXECUTED DRAFT requiring activation outputs that do not yet exist.

Summarize the fixed pilot; all uncertainty resamples whole prompt pairs.
Do not run locally under the current project constraint.
"""
import csv
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'
os.environ['MPLCONFIGDIR'] = str(ROOT/'.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def main():
    rows = [json.loads(line) for line in (OUT/'scores.jsonl').read_text().splitlines()]
    protocol = json.loads((OUT/'protocol.json').read_text())
    pairs = sorted(set(r['pair_id'] for r in rows))
    by_id = {(r['pair_id'],r['label']):r for r in rows}
    assert len(rows) == len(by_id) == 2*len(pairs)
    assert len(pairs) == protocol['n_pairs'], 'Do not present incomplete pilot as complete'
    rows = [by_id[(p,label)] for p in pairs for label in [0,1]]
    y = np.array([r['label'] for r in rows])
    names = {'projection':'Raw activation projection', 'delta_projection':'Activation projection difference',
             'llm_score':'Qwen text judgement', 'lexical_score':'Fixed caveat count',
             'tfidf_score':'TF-IDF text classifier', 'response_tokens':'Answer length'}
    rng = np.random.default_rng(protocol['seed'])
    draws = rng.integers(0,len(pairs),(2000,len(pairs)))
    metrics = {}
    for key, name in names.items():
        values = np.array([np.nan if r[key] is None else r[key] for r in rows],dtype=float).reshape(-1,2)
        valid = np.isfinite(values).all(axis=1)
        v = values[valid]
        labels = np.tile([0,1],len(v))
        boots = v[rng.integers(0,len(v),(2000,len(v)))] if not valid.all() else v[draws]
        aucs = np.array([roc_auc_score(labels,b.reshape(-1)) for b in boots])
        gaps = (boots[:,:,1]-boots[:,:,0]).mean(axis=1)
        metrics[key] = dict(name=name,n_pairs=int(valid.sum()),missing_rows=int(np.isnan(values).sum()),
            mean_normal=float(v[:,0].mean()), mean_hallucination_II=float(v[:,1].mean()),
            auc=float(roc_auc_score(labels,v.reshape(-1))),auc_ci=np.quantile(aucs,[.025,.975]).tolist(),
            paired_gap=float((v[:,1]-v[:,0]).mean()),paired_gap_ci=np.quantile(gaps,[.025,.975]).tolist(),
            paired_win_rate=float(((v[:,1]>v[:,0])+.5*(v[:,1]==v[:,0])).mean()))
    delta = np.array([r['delta_projection'] for r in rows])
    correlations = {}
    top_k = len(rows)//4
    top_delta = set(np.argsort(delta)[-top_k:])
    for key in ['llm_score','lexical_score','tfidf_score','response_tokens','projection','unfiltered_delta']:
        x = np.array([np.nan if r[key] is None else r[key] for r in rows],dtype=float)
        valid = np.isfinite(x)
        correlations[key] = dict(
            spearman=float(spearmanr(delta[valid],x[valid]).statistic),
            within_normal=float(spearmanr(delta[valid & (y==0)],x[valid & (y==0)]).statistic),
            within_hallucination_II=float(spearmanr(delta[valid & (y==1)],x[valid & (y==1)]).statistic),
            top_quartile_overlap=len(top_delta & set(np.argsort(np.nan_to_num(x,nan=-np.inf))[-top_k:])),
            top_quartile_size=top_k)
    vectors = json.loads((OUT/'vector_manifest.json').read_text())
    naturals = [json.loads(line) for line in (OUT/'natural_responses.jsonl').read_text().splitlines()]
    order = np.argsort(delta)
    groups = {}
    for name,idx in [('low',order[:top_k]),('high',order[-top_k:])]:
        groups[name] = dict(n=len(idx),mean_delta=float(delta[idx].mean()),
            fraction_hallucination_II=float(y[idx].mean()))
        (OUT/f'{name}_projection_subset.jsonl').write_text(''.join(json.dumps(rows[i])+'\n' for i in idx))
    summary = dict(n_pairs=len(pairs),n_examples=len(rows),metrics=metrics,
        correlations=correlations,selected_quartiles=groups,vector=vectors,
        truncated_natural_responses=sum(r['finish_reason']=='length' for r in naturals),
        uncertainty='2000 prompt-pair bootstrap resamples; conditional on this one extracted vector and fitted text classifier.',
        scope='Data-label separation and ranking agreement only. No SFT outcomes measured.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (OUT/'scores.csv').open('w') as f:
        keys = [k for k in rows[0] if k != 'messages']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titlesize':11})
    fig,ax=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    colors=['#2171a6','#d65c3b']
    bins=np.linspace(delta.min()-.2,delta.max()+.2,19)
    for label,name,color in zip([0,1],['Normal','Hallucination II'],colors):
        ax[0].hist(delta[y==label],bins=bins,alpha=.65,color=color,label=name)
    ax[0].set(xlabel='Response projection − natural-answer projection',ylabel='Examples',title='A  Published dataset variants')
    ax[0].legend(frameon=False,fontsize=9)
    for label,color in enumerate(colors):
        idx=y==label
        ax[1].scatter(delta[idx],[r['tfidf_score'] for r in np.array(rows,dtype=object)[idx]],
            s=19,alpha=.7,color=color,edgecolors='none')
    rho=correlations['tfidf_score']['spearman']
    ax[1].set(xlabel='Activation projection difference',ylabel='Text classifier score',
        title=f'B  Text versus activations (ρ = {rho:.2f})')
    keys=['delta_projection','llm_score','lexical_score','tfidf_score','response_tokens']
    labels=['Activation ΔP','Qwen text judge','Caveat count','TF-IDF text','Answer length']
    for i,k in enumerate(keys):
        m=metrics[k];a=m['auc'];lo,hi=m['auc_ci']
        ax[2].errorbar(a,i,xerr=[[max(0,a-lo)],[max(0,hi-a)]],fmt='o',color='#26364a',capsize=3)
        ax[2].text(.515,i+.20,f'{a:.3f}',fontsize=8,color='#505967')
    ax[2].set(yticks=range(len(keys)),yticklabels=labels,xlim=(.50,1.025),
        xlabel='AUC for dataset label',title='C  Both methods see the supplied text')
    ax[2].invert_yaxis();ax[2].grid(axis='x',alpha=.15)
    fig.suptitle('Hallucination data screening • Qwen2.5-7B 4-bit • 64 matched prompt pairs',fontsize=13,fontweight='bold')
    fig.savefig(OUT/'comparison.png',dpi=180)
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
