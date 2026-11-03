from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import json,shutil
p=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(7.15,3.25)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100);ax.set_ylim(0,47);ax.axis('off')
blue='#27799c';red='#b64b43';gray='#929aa1'
def txt(x,y,t,size=7,**kw):ax.text(x,y,t,ha='center',va='center',fontsize=size,**kw)
def box(x,y,w,h,color=gray,fill=False,lw=.8,ls='-'):
 ax.add_patch(Rectangle((x,y),w,h,facecolor=color if fill else 'none',edgecolor=color,lw=lw,ls=ls,alpha=.18 if fill else 1))
def base(x,y,w=17,h=11):
 box(x,y,w,h);box(x+2,y+2,4,4,'#d5dde0',True);box(x+9,y+1,6,7,'#d5dde0',True)
 txt(x,y+h+1,'(0,0)',5,ha='left') if False else None
 return w,h
def grid(x,y,w,h,n,m,color=blue):
 for i in range(n):
  for j in range(m):box(x+i*(w-h/m)/(n-1) if n>1 else x,y+j*(h-h/m)/(m-1) if m>1 else y,h/m,h/m,color,lw=.55)
def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=8,lw=.8,color=gray))
# Common source rectangle. Top row shows sampled regions in ORIGINAL coordinates.
for x,title in [(2,'Native center crop'),(27,'grid9'),(52,'fov3'),(77,'Whole-image resize')]:
 txt(x+9,44,title,8,weight='bold');base(x,29)
 txt(x+8.5,27.2,'Original coordinates: W × H',6)
box(9,33,3,3,blue,lw=1.4); txt(10.5,24.5,'200 × 200 → n; 1 view',6.8)
for i in range(3):
 for j in range(3):box(27+i*7,29+j*4,3,3,blue)
txt(35.5,24.5,'Native windows → n; ≤9 views',6.8)
for i in range(3):
 for j in range(3):box(52+i*(17-11/3)/2,29+j*(11-11/3)/2,11/3,11/3,blue)
txt(60.5,24.5,'AREA to short side 600; ≤9 views',6.5)
box(77,29,17,11,blue,lw=1.4);txt(85.5,24.5,'AREA to n × n; 1 view',6.8)
txt(50,21.8,'Blue boxes show source support; crop windows use CUBIC to canonical size n.',7)
# Bottom row two source frames per condition.
for x,title in [(2,'Coverage: 25% → 100%'),(35,'Downscale: k=1 → k=2'),(68,'Round trip: k=1 → 2 → 1')]:
 txt(x+14,18.8,title,8,weight='bold');base(x,5,12,9);base(x+17,5,12,9);arrow(x+12.5,9.5,x+16.5,9.5)
box(5,7.25,6,4.5,blue);grid(5,7.25,6,4.5,3,2);grid(19,5,12,9,6,4)
txt(16,2.2,'Same native window; more windows',6.8)
grid(35,5,12,9,6,4);grid(52,5,12,9,3,2)
txt(49,2.2,'Fixed source area; AREA; ~4× fewer windows',6.5)
grid(68,5,12,9,6,4);grid(85,5,12,9,6,4,red)
txt(82,2.2,'AREA down + CUBIC back; same crop grid',6.5)
fig.savefig(p/'fig_preprocessing_audited.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)
# Recreate the factorial figure using only recorded JSON point estimates.
fig,axs=plt.subplots(1,4,figsize=(7.15,2.0),sharex=True,sharey=True)
for j,(dataset,br) in enumerate([('NTIRE','A2'),('NTIRE','B2swa'),('SocialRF','A2'),('SocialRF','B2swa')]):
 f='test14_tile-factorial.json' if dataset=='NTIRE' else 'test16_tile-factorial-socialrf.json'
 d=json.loads(((Path(__file__).resolve().parent/'evidence/docs/ntire/amend5')/f).read_text())['branches'][br]['pooled'];a=axs[j]
 for k,col in [(1,'#ad523e'),(2,blue)]:
  a.plot([.25,.5,1],[d[f'robust_tiles_k{k}_f{x}']['point'] for x in ['0.25','0.5','1']],'-o',color=col,ms=3,label=f'{k}× downscale')
 a.plot([1],[d['robust_tiles_k4_f1']['point']],'^',color='#39834b',ms=5,label='4× (full only)')
 a.axhline(d['robust_squish']['point'],color='black',ls='--',lw=.8,label='Whole image')
 a.axhline(d['robust_center']['point'],color='0.45',ls=':',lw=.8,label='Center crop')
 a.set_title(f'{dataset}, {"A" if br=="A2" else "B"}',fontsize=8);a.set_ylim(.85,1);a.set_xticks([.25,.5,1],['25','50','100']);a.set_xlabel('Image area covered (%)',fontsize=7);a.tick_params(labelsize=7);a.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Robust ROC AUC',fontsize=8)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=6.5,frameon=False,bbox_to_anchor=(.51,-.02));fig.subplots_adjust(left=.075,right=.99,top=.88,bottom=.3,wspace=.22)
fig.savefig(p/'fig_tiles_audited.pdf',bbox_inches='tight',pad_inches=.02);plt.close(fig)
d=json.loads((Path(__file__).resolve().parent/'evidence/docs/ntire/paper/analysis_figsize.json').read_text())['bins']
fig,a=plt.subplots(figsize=(3.4,2.0));a.axvspan(384,512,color='0.90',lw=0)
x=[v['median'] for v in d]
for key,col,label in [('crop','#ad523e','Crop-trained'),('squish',blue,'Whole-image-trained')]:
 y=[v[key] for v in d];lo=[v[key+'_ci'][0] for v in d];hi=[v[key+'_ci'][1] for v in d]
 a.plot(x,y,'o-',color=col,ms=3,label=label);a.fill_between(x,lo,hi,color=col,alpha=.13,lw=0)
a.set_xscale('log');a.set_xticks([150,300,500,1000,2000],['150','300','500','1000','2000']);a.set_xlabel('Native short side (pixels)',fontsize=8);a.set_ylabel('Robust ROC AUC',fontsize=8);a.tick_params(labelsize=7);a.spines[['top','right']].set_visible(False);a.legend(fontsize=6.5,frameon=False,loc='lower left');fig.tight_layout(pad=.5);fig.savefig(p/'fig_size_audited.pdf',bbox_inches='tight',pad_inches=.02);plt.close(fig)
