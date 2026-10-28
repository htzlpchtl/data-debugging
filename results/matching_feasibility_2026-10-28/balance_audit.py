"""Post-hoc exact-token-length sensitivity analysis; no new API/model calls."""
import json,pathlib,shutil,sys
root=pathlib.Path('/workspace/data-debugging')
sys.path.insert(0,str(root/'remote'))
from run import remote_only,read_rows,write_json,sha
remote_only()
from matching_feasibility import analyze
parent=root/'runs/matching-feasibility-2026-10-28'
out=parent/'length_exact';out.mkdir(exist_ok=True)
cfg=json.loads((root/'remote/matching-feasibility-2026-10-28.json').read_text())
cfg['maximum_response_length_ratio']=1.0
for name in ('judgements.jsonl','judge_usage.jsonl'):shutil.copyfile(parent/name,out/name)
write_json(out/'analysis_protocol.json',{'status':'POST_HOC_SENSITIVITY_NO_NEW_SFT','motivation':'Original 20% length caliper produced primary activation-case length SMD 0.674. Require exact token counts to remove that observed imbalance.','parent_manifest_sha256':sha(parent/'manifest.json'),'config':cfg})
analyze(out,cfg,read_rows(parent/'inputs.jsonl'))
