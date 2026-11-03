"""Verify the submission against packaged reports only; never loads images or models."""
from pathlib import Path
import json,re,hashlib
R=Path(__file__).resolve().parent
P=R.parent

def read(name):
    return json.loads((R/'evidence'/name).read_text())

def table(label):
    s=(P/'main_reframe_audited.tex').read_text(); end=s.index('\\label{'+label+'}')
    a=s.rfind('\\begin{table',0,end)
    return s[a:end]

checks=[]
def check(name,actual,expected,tol=1e-10):
    assert abs(actual-expected)<=tol,(name,actual,expected)
    checks.append({'check':name,'actual':actual,'expected':expected,'status':'pass'})
def printed(name,v,decimals,where):
    t=f'{abs(v):.{decimals}f}'
    assert t in where,(name,t)
    checks.append({'check':name,'printed':t,'status':'pass'})

# Every robust/clean AUC in the frozen presentation table.
for b in ['A2','B2swa']:
    d=read(f'docs/ntire/crop_probe_{b}.json')['arms']
    for a in ['center','grid9','fov3','squish']:
        for m in ['robust_auc','clean_auc']:printed(f'probe {b}/{a}/{m}',d[a][m],4,table('tab:probe'))
# Trained branch table, separate pass.
d=read('docs/ntire/paper/paper_results.json')
for b in ['A','B']:
    for arm in ['crop','crop_testtime_squish','trained_squish']:
        for m in ['robust_auc','clean_auc']:printed(f'branch {b}/{arm}/{m}',d['tables']['branch_'+b]['models'][arm][m]['value'],4,table('tab:branches'))
# External pooled AUCs: both all-distributed and JPEG 95 rows.
for ds in ['aigi_socialrf','aigi_community','synthbuster']:
    for suffix in ['', '_jpeg95']:
        for arm in ['crop','squish']:
            v=read(f'docs/ntire/paper/fusion_{arm}.json')['splits']['ext_'+ds+suffix]['fused']['auc']
            printed(f'external {ds}{suffix}/{arm}',v,4,table('tab:external'))
    for suffix,dec in [('',3),('_deg',3),('_jpeg95',4)]:
        v=d['tables']['ext_'+ds+suffix]['vs_baseline']['squish']['auc']
        for k in v['ci']:printed(f'external interval {ds}{suffix}',k,dec,table('tab:external'))
# B-Free reference and resized presentations, with subgroup status.
bf=read('docs/ntire/paper/public_bfree.json')['splits']
for ds,m in [('ntire_val_hard','robust_auc'),('ntire_val_hard','clean_auc'),('ntire_val','robust_auc'),('ntire_val','clean_auc')]+[('ext_'+q+s,'auc') for q in ['aigi_socialrf','aigi_community','synthbuster'] for s in ['','_deg']]:
    for a in ['native','train_fov','squish']:printed(f'B-Free {ds}/{m}/{a}',bf[ds][a][m],3,table('tab:public'))
# Exact SocialRF headline, early coverage gain, and factorial scope.
t=read('docs/ntire/amend5/test16_tile-factorial-socialrf.json')
main=(P/'main_reframe_audited.tex').read_text();supp=(P/'supplement_reframe_audited.tex').read_text()
for b,lo,hi,early in [('A2',.867656,.958200,5.3364),('B2swa',.882892,.962304,5.0968)]:
    x=t['branches'][b]['pooled'];check(b+' center',x['robust_center']['point'],lo);check(b+' whole',x['robust_squish']['point'],hi)
    check(b+' early coverage in AUC points',100*(x['robust_tiles_k1_f0.25']['point']-lo),early)
    printed(b+' headline before',lo,6,main);printed(b+' headline after',hi,6,main)
    check(b+' degraded n',x['n_distorted'],1000)
    for k in ['robust_dF','robust_dK']:printed(b+'/'+k,x[k]['point'],3,table('tab:decomp'))
# Round-trip values and ordering intervals; original dS is renamed dV.
rt=read('docs/ntire/amend5/test17_roundtrip.json')
for ds in ['ntire','socialrf']:
    for b in ['A2','B2swa']:
        x=rt['row_sets'][ds][b]['pooled']
        for k in ['robust_dA','robust_dS']:printed(f'{ds}/{b}/{k}',x[k]['point'],3,table('tab:decomp'))
o=read('docs/ntire/amend5/test18_shrinkorder.json')
for b in ['A2','B2swa']:
    x=o['branches'][b]['stats'];check(b+' ordering source n',x['n'],2833)
    for k in ['reduction_k2','reduction_k4']:
        for v in x[k]['ci']:printed(b+'/'+k+' interval',v,3,main)
# Frozen in-band result and exact native-crop population/contrasts.
x=read('docs/ntire/amend5/test12_writing.json')['bin_416_496'];check('in-band n',x['n'],171)
for b in ['A','B']:
    v=x['branches'][b]['squish at test'];printed(b+' in-band points',v['diff']*100,1,main)
    for c in v['ci']:printed(b+' in-band interval points',c*100,1,main)
x=read('docs/ntire/amend5/test19_stretch.json')['splits']['ntire_val_hard']['short_ge_512']['robust']
check('native control n',x['n'],630);check('native control real n',x['n_real'],320);check('native control fake n',x['n_fake'],310)
for k,expected in [('E',.01148185483870967),('V',.0478326612903226),('T',.05931451612903227)]:
    check('native contrast '+k,x[k]['diff'],expected)
    for v in [x[k]['diff']]+x[k]['ci']:printed('native '+k,v,3,main)
check('native contrast identity',x['T']['diff'],x['V']['diff']+x['E']['diff'])
# Distinguish all four fusion configurations.
for fn,expected,pop in [('docs/ntire/paper/fusion_crop.json',.5,'split=ntire_val'),('docs/ntire/fusion_squish_valfit.json',.5,'split=ntire_val'),('docs/ntire/paper/fusion_squish.json',.55,'split=ntire_val'),('docs/ntire/fusion_ntire2_final.json',.45,'split=ntire_val+ntire_val_hard')]:
    f=read(fn)['fusion'];check(fn+' weight',f['weight'],expected);assert f['fit_population']==pop
# Saved historical test reports only, no test rows or scores.
for fn in ['docs/ntire/fusion_ntire2_final.json','docs/ntire/paper/fusion_squish_final.json']:
    v=read(fn)['splits']['ntire_test']['fused']
    for k in ['robust_auc','clean_auc','auc']:printed(fn+'/'+k,v[k],4,table('tab:test'))
    check(fn+' distorted share',v['n_robust']/v['n'],.5)
# Source-preservation evidence for packaged snapshots.
for x in json.loads((R/'evidence_ledger.json').read_text()):
    f=P/x['packaged'];assert hashlib.sha256(f.read_bytes()).hexdigest()==x['packaged_sha256'],f
for tex in [main,supp]:
    assert '\\bibliography{refs_reframe_audited}' in tex
    assert not re.search(r'\\pending\{|UNCONFIRMED',tex)
assert 'branch B only' in supp and 'training-size mismatch' in main.lower()
out={'boundary':'Saved JSON/Markdown verification only; no inference, training, per-image scoring, or test evaluation.','n_checks':len(checks),'checks':checks}
(R/'numeric_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(f'PASS: {len(checks)} numerical/presentation checks and 121 evidence hashes; no models or image data loaded.')
