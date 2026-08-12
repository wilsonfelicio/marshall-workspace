import warnings, sys; warnings.filterwarnings('ignore')
import json, numpy as np, pandas as pd, textwrap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

slug=sys.argv[1] if len(sys.argv)>1 else 'aguacate'
D=json.loads(open(f'charts/{slug}.html').read().split('const D=',1)[1].split(';\nconst dark',1)[0])
MES={1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}
sp=lambda d: f"{MES[d.month]} {d.year}"

mes=pd.PeriodIndex(D['meses'],freq='M').to_timestamp()
g=lambda k: pd.Series([np.nan if v is None else v for v in D[k]],index=mes)
wy,ry,gap=g('wy'),g('ry'),g('gap')
ok=wy.notna()&ry.notna()
lead=100*(gap[ok]>0).mean()
# amplitude on LOG changes: percent changes are asymmetric for large moves and
# these series reach +345%, which biases the comparison.
wl=pd.Series([np.nan if v is None else v for v in D['series'][0]['vals']],index=mes)
rl=pd.Series([np.nan if v is None else v for v in D['retail']],index=mes)
amp=float((np.log(wl)-np.log(wl.shift(12)))[ok].std()/(np.log(rl)-np.log(rl.shift(12)))[ok].std())
corr=D['stats']['corr_mom']; last=wy.last_valid_index(); first=wy.first_valid_index()
nm=D['generico'].split(' ',1)[1] if ' ' in D['generico'] else D['generico']

BLUE,ORANGE,RED='#2a78d6','#eb6834','#e34948'
SURF,INK,SEC,MUT,GRID,AXIS,ZERO='#fcfcfb','#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7','#898781'
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':SURF,'axes.facecolor':SURF,
  'savefig.facecolor':SURF,'text.color':INK,'xtick.color':MUT,'ytick.color':MUT,
  'axes.edgecolor':AXIS,'axes.linewidth':0.9,'xtick.labelsize':10.5,'ytick.labelsize':10.5})

fig=plt.figure(figsize=(12.8,9.0)); L,Rt=0.078,0.845
ax =fig.add_axes([L,0.325,Rt-L,0.465])
ax2=fig.add_axes([L,0.112,Rt-L,0.170],sharex=ax)
for a in (ax,ax2):
    a.grid(axis='y',color=GRID,lw=0.9); a.set_axisbelow(True)
    for s in ('top','right'): a.spines[s].set_visible(False)
    a.axhline(0,color=ZERO,lw=1.3,zorder=3)

ax.plot(wy.index,wy.values,color=BLUE,lw=1.7,zorder=5)
ax.plot(ry.index,ry.values,color=ORANGE,lw=1.7,zorder=4)
ax.set_ylabel('variación anual',fontsize=10.5,color=SEC,labelpad=10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{'+' if v>0 else ''}{v:.0f}%"))
plt.setp(ax.get_xticklabels(),visible=False)

span=ax.get_ylim()[1]-ax.get_ylim()[0]
pts=sorted([[wy[last],BLUE,'Mayorista',wy[last]],[ry[last],ORANGE,'Menudeo',ry[last]]],key=lambda x:-x[0])
if pts[0][0]-pts[1][0] < span*0.05:
    mid=(pts[0][0]+pts[1][0])/2; pts[0][0]=mid+span*0.028; pts[1][0]=mid-span*0.028
for ypos,c,n,vt in pts:
    ax.annotate(f"{n} {vt:+.0f}%",xy=(last,ypos),xytext=(9,0),textcoords='offset points',
                color=c,fontsize=11,fontweight='bold',va='center',annotation_clip=False)

ax2.fill_between(gap.index,0,gap.clip(lower=0).values,color=BLUE,alpha=0.55,lw=0,zorder=2)
ax2.fill_between(gap.index,0,gap.clip(upper=0).values,color=RED,alpha=0.55,lw=0,zorder=2)
ax2.plot(gap.index,gap.values,color=INK,lw=0.8,alpha=0.4,zorder=4)
ax2.set_ylabel('brecha, pp',fontsize=10.5,color=SEC,labelpad=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{'+' if v>0 else ''}{v:.0f}"))
m=np.nanmax(np.abs(gap.values))*1.1; ax2.set_ylim(-m,m)
ax2.text(0.008,0.88,'mayorista sube más rápido',transform=ax2.transAxes,color=BLUE,fontsize=9.5,fontweight='bold',va='top')
ax2.text(0.008,0.12,'menudeo sube más rápido',transform=ax2.transAxes,color=RED,fontsize=9.5,fontweight='bold',va='bottom')

fig.text(L,0.968,f'{nm}: variación anual del precio mayorista y del INPC',
         fontsize=16,fontweight='bold',color=INK)
sub=(f'Cambio % respecto al mismo mes del año anterior. Mayorista: índice Jevons de SNIIM. '
     f'Menudeo: INPC genérico {D["generico"]} (INEGI). '
     f'{int(ok.sum())} meses, {sp(first)} – {sp(last)}. Correlación de cambios mensuales {corr:.3f}; '
     f'amplitud del mayorista {amp:.2f}× la del menudeo (cambios log); '
     f'por delante en {lead:.0f}% de los meses.')
y=0.934
for ln in textwrap.wrap(sub,width=118):
    fig.text(L,y,ln,fontsize=10.8,color=SEC); y-=0.0255
fig.legend(handles=[Line2D([],[],color=BLUE,lw=2.2,label='Mayorista SNIIM'),
                    Line2D([],[],color=ORANGE,lw=2.2,label='Menudeo INPC')],
           loc='upper left',bbox_to_anchor=(L,y+0.012),frameon=False,ncol=2,
           fontsize=10.8,handlelength=1.7,columnspacing=2.2,labelcolor=SEC)

foot=('Fuente: SNIIM (Secretaría de Economía) e INEGI. El mes en curso se excluye por estar incompleto. '
      'Las ventanas anuales se traslapan 11 de 12 meses, así que estas series están fuertemente '
      'autocorrelacionadas: la cifra utilizable para medir la relación es la correlación de los cambios '
      'mensuales, no la de las series anuales.')
for i,ln in enumerate(textwrap.wrap(foot,width=150)):
    fig.text(L,0.050-i*0.020,ln,fontsize=9,color=MUT)

out=f'charts/{slug}_yoy.png'; fig.savefig(out,dpi=170)
print(f"{out}  n={int(ok.sum())} corr={corr} amp={amp:.2f} lead={lead:.0f}%  last {wy[last]:+.1f}% / {ry[last]:+.1f}%")
