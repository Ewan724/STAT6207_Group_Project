#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))
from src.config import load_config, add_common_args

def norm(s):
    s = str(s or '').lower().strip()
    s = re.sub(r"[^a-z0-9+#./ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    parser = argparse.ArgumentParser(); add_common_args(parser); args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg['_project_root']); out = Path(cfg['_out_dir']); proc = out/'data'/'processed'; tables = out/'results'/'tables'
    tables.mkdir(parents=True, exist_ok=True)
    from src.data import load_excel
    _, cv_xls = load_excel(root / cfg['paths']['xlsx'])
    cv_xls['Ref']=cv_xls['Ref'].astype(str).str.strip()
    cv_ext = {d['doc_id']: d for d in [json.loads(l) for l in open(proc/'cv_extracted.jsonl', encoding='utf-8')]}
    rows=[]
    for _, r in cv_xls.iterrows():
        ref=r['Ref']; ext=cv_ext.get(ref,{})
        excel_skills = str(r.get('Skills','') or '')
        ex_set = set([norm(x) for x in re.split(r',|;|\n|\|', excel_skills) if norm(x)])
        llm_set = set([norm(x) for x in (ext.get('skills',[]) or []) + (ext.get('tools_technologies',[]) or []) if norm(x)])
        inter = len(ex_set & llm_set)
        prec = inter/len(llm_set) if llm_set else 0.0
        rec = inter/len(ex_set) if ex_set else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        rows.append({'ref':ref,'excel_count':len(ex_set),'llm_count':len(llm_set),'precision':prec,'recall':rec,'f1':f1})
    df = pd.DataFrame(rows)
    df.to_csv(proc/'extraction_quality.csv', index=False)
    summary = {'macro_precision':float(df['precision'].mean()), 'macro_recall':float(df['recall'].mean()), 'macro_f1':float(df['f1'].mean()), 'n':len(df)}
    pd.DataFrame([summary]).to_csv(tables/'extraction_quality_summary.csv', index=False)
    df.sort_values('f1').head(5).to_csv(tables/'extraction_quality_worst5.csv', index=False)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
