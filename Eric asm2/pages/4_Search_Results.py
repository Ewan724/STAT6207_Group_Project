import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Search Results", layout="wide")
OUT = Path(__file__).resolve().parents[1]
PROC = OUT/"data"/"processed"

st.header("Search Results (Top 10 per JD)")
if not (PROC/"top10.csv").exists() or not (PROC/"ground_truth.csv").exists():
    st.warning("Rankings not found. Run stages 4-5.")
    st.stop()

top10 = pd.read_csv(PROC/"top10.csv")
rankings = pd.read_csv(PROC/"rankings.csv") if (PROC/"rankings.csv").exists() else None
gt = pd.read_csv(PROC/"ground_truth.csv")
jd_ids = gt['jd_id'].tolist()
models = sorted(top10['model'].unique().tolist())

c1,c2 = st.columns(2)
jd_id = c1.selectbox("JD", jd_ids)
model = c2.selectbox("Model", models, index=models.index('M3_ae32') if 'M3_ae32' in models else 0)

g = gt[gt.jd_id==jd_id].iloc[0]
st.subheader(f"{jd_id}: {g['jd_title']}")
st.write(f"Role family: {g['role_family']} | Location: {g.get('jd_location','')} | Salary: {g.get('jd_salary','')}")
st.caption(f"Ground truth hired CV: {g['hired_cv_id']} — {g['hired_cv_title']} ({g['hired_cv_company']})")

sub = top10[(top10.jd_id==jd_id)&(top10.model==model)].sort_values('rank')
hired_id = g['hired_cv_id']
# highlight
def highlight(row):
    if row['cv_id']==hired_id:
        return ['background-color: #d4edda']*len(row)
    return ['']*len(row)
st.dataframe(sub.style.apply(highlight, axis=1), use_container_width=True)

# hired rank
if rankings is not None:
    hr = rankings[(rankings.jd_id==jd_id)&(rankings.model==model)&(rankings.cv_id==hired_id)]
    if len(hr):
        rk = int(hr.iloc[0]['rank']); sc = float(hr.iloc[0]['score'])
        if rk>10:
            st.error(f"Hired CV ranked #{rk} with score {sc:.4f} (outside top 10).")
        else:
            st.success(f"Hired CV ranked #{rk} with score {sc:.4f}.")

st.subheader("Candidate details")
sel = st.selectbox("Select candidate from top 10", sub['cv_id'].tolist())
row = sub[sub.cv_id==sel].iloc[0]
st.write(f"**{row['cv_id']}** — {row.get('cv_title','')} @ {row.get('cv_company','')} ({row.get('role_family','')})")
st.write(f"Score: {row['score']:.4f} | Relevant (ground truth): {'Yes' if row['is_relevant'] else 'No'}")
st.write(f"Must-have matches: {row.get('must_have_matches','')}")
st.write(f"Missing: {row.get('missing_must_haves','')}")
st.info(row.get('reason',''))

st.download_button("Download top-10 CSV", data=sub.to_csv(index=False), file_name=f"top10_{jd_id}_{model}.csv", mime="text/csv")
if rankings is not None:
    full = rankings[(rankings.jd_id==jd_id)&(rankings.model==model)].sort_values('rank')
    st.download_button("Download full ranking CSV", data=full.to_csv(index=False), file_name=f"ranking_{jd_id}_{model}.csv", mime="text/csv")
st.caption("Baseline does not apply salary/location/notice hard filters.")
