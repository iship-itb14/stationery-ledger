"""Real-data analysis: Amazon Reviews 2023 (Office Products + Arts/Crafts/Sewing metadata, 60k review sample)."""
import re, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
rng=np.random.default_rng(0)
M=pd.read_csv("data/stationery_meta_small.csv"); V=pd.read_csv("data/stationery_reviews_small.csv")

# ---------- 1. clean + classify ----------
M["brand"]=M.brand.fillna("").str.replace(r"^(Visit the |Brand: )|( Store)$","",regex=True).str.strip()
low=(M.title.fillna("")+" ").str.lower()
RULES=[("Accessory",r"pencil case|pen case|pouch|holder|grips?\b|sharpener|cover|stand\b|organizer|refill pack for|stamp|stencil|bag\b|box\b"),
("Ink",r"bottled ink|ink bottle|ink set|calligraphy ink|drawing ink|fountain pen ink|india ink|ink cartridge"),
("Highlighter",r"highlighter"),("Washi tape",r"washi"),("Sticky notes",r"sticky note|post-it|memo pad|index tab"),
("Eraser",r"eraser"),("Mechanical pencil",r"mechanical pencil|lead pencil"),("Colored pencil",r"colou?red pencil|pencils? set"),
("Pencil",r"pencil"),("Brush pen",r"brush pen|brush marker"),("Gel pen",r"\bgel\b"),("Ballpoint",r"ballpoint|ball point|ball pen"),
("Fountain pen",r"fountain"),("Marker",r"marker"),("Fineliner",r"fine ?liner|felt tip|micron|drawing pen|technical pen"),
("Sketchbook",r"sketch ?book|sketch pad|drawing pad"),("Planner",r"planner|calendar|agenda"),
("Notebook",r"notebook|journal|composition|diary|notepad|note pad"),("Pen (other)",r"\bpens?\b")]
def cls(t):
    for n,p in RULES:
        if re.search(p,t): return n
    return "Other"
M["type"]=low.map(cls)
M=M[~M.type.isin(["Other","Accessory"])]
def pack(t):
    m=re.search(r"(\d{1,3})\s*[- ]?(?:pack|pcs|pc|pieces|count|ct|pens|pencils|markers|highlighters|erasers|colors|colours|books|notebooks|journals)\b",t)
    return int(m.group(1)) if m and 1<=int(m.group(1))<=300 else 1
M["pack"]=low.loc[M.index].map(pack)
M=M[(M.price>=0.5)&(M.price<=200)].sort_values("rating_number",ascending=False).drop_duplicates(["title","brand"])
M["unit_price"]=M.price/M.pack
b=M.brand.str.lower().str.replace(r"[^a-z0-9]","",regex=True).replace({"uni":"uniball","mitsubishipencil":"uniball","mitsubishi":"uniball"})
disp=M.groupby(b).brand.agg(lambda s:s.value_counts().index[0]); M["bkey"]=b; M["brand"]=b.map(disp)
M=M[M.brand!=""]; print("products after cleaning:",len(M)); print(M.type.value_counts().to_string())
C=M.rating_number.median(); M["bayes"]=(M.rating_number*M.average_rating+ 50*M.average_rating.mean())/(M.rating_number+50)
R={"n_products":len(M),"n_brands":int(M.brand.nunique()),"n_reviews":len(V),"median_price":round(M.price.median(),2),"mean_rating":round(M.average_rating.mean(),2),
   "share_4plus":round((M.average_rating>=4).mean(),3)}

# ---------- 2. statistics ----------
rho,p=stats.spearmanr(M.price,M.average_rating); R["price_rating"]={"rho":round(rho,3),"p":float(p)}
w=[(t,*stats.spearmanr(g.price,g.average_rating)) for t,g in M.groupby("type") if len(g)>=200]
R["price_rating_by_type"]={t:round(r,3) for t,r,_ in w}
print("overall rho",round(rho,3),"by type",R["price_rating_by_type"])
# pack size discount: log(unit price) ~ log(pack) within type
disc={}
for t,g in M.groupby("type"):
    g=g[g.pack>1]
    if len(g)>=150:
        s=stats.linregress(np.log(g.pack),np.log(g.unit_price)); disc[t]=round((2**s.slope-1)*100,1)
R["pack_discount_per_doubling"]=disc; print("unit price change when pack doubles (%):",disc)
JP={"pilot","uniball","zebra","pentel","sakura","tombow","sailor","platinum","kuretake","muji","kokuyo","pigma","yasutomo"}
M["jp"]=M.bkey.isin(JP); M["res"]=M.average_rating-M.groupby("type").average_rating.transform("mean")
a,bb=M[M.jp].res,M[~M.jp].res; u,pj=stats.mannwhitneyu(a,bb)
R["japan"]={"n_jp":int(M.jp.sum()),"diff":round(a.mean()-bb.mean(),3),"p":float(pj)}; print("JP diff",R["japan"])
# ratings are ceiling-bound: how much information is there?
R["rating_dist"]=M.average_rating.round().value_counts().sort_index().to_dict()

# ---------- 3. brands ----------
BR=[]
for (t,br),g in M.groupby(["type","brand"]):
    if len(g)>=3: BR.append([t,br,len(g),round(np.average(g.average_rating,weights=g.rating_number),3),round(g.price.median(),2),int(g.rating_number.sum())])
R["brand_rows"]=BR
tb=M.groupby("brand").agg(n=("brand","size"),r=("average_rating","mean"),p=("price","median"),rev=("rating_number","sum")).query("n>=25").sort_values("rev",ascending=False).head(12)
print(tb.round(2))

# ---------- 4. reviews ----------
V=V.merge(M[["parent_asin","type"]],on="parent_asin"); V["text"]=(V.title.fillna("")+". "+V.text.fillna("")).str.lower()
V["dt"]=pd.to_datetime(V.timestamp,unit="ms"); V["m"]=V.dt.dt.month; V["y"]=V.dt.dt.year
ASP={"Ink bleeds or ghosts":r"bleed|ghost|show.?through|see.?through","Smudges":r"smudg|smear","Dries out / dies fast":r"dried? out|dry out|stopped working|ran out|died|dead",
"Leaks":r"leak","Breaks":r"broke|broken|snap|crack|fell apart","Scratchy or skips":r"scratch|skip|gritty|rough","Smooth writing":r"smooth|glide|effortless","Cheap or flimsy":r"cheap|flimsy|thin|poor quality"}
asp={}
for t,g in V.groupby("type"):
    if len(g)<800: continue
    neg=g[g.rating<=2]; pos=g[g.rating>=4]
    asp[t]={"n":len(g),"neg_n":len(neg),"neg":{k:round(neg.text.str.contains(r).mean()*100,1) for k,r in ASP.items()},"pos":{k:round(pos.text.str.contains(r).mean()*100,1) for k,r in ASP.items()}}
R["aspects"]=asp
ok=V[V.rating!=3].copy(); ok["y"]=(ok.rating<=2).astype(int); ok=ok.sample(min(40000,len(ok)),random_state=0)
tf=TfidfVectorizer(min_df=15,ngram_range=(1,2),stop_words="english",max_features=20000); X=tf.fit_transform(ok.text)
lr=LogisticRegression(C=1,max_iter=300,class_weight="balanced").fit(X,ok.y); cv=cross_val_score(lr,X,ok.y,cv=3,scoring="roc_auc")
f=np.array(tf.get_feature_names_out()); o=np.argsort(lr.coef_[0])
R["text_model"]={"auc":round(cv.mean(),3),"complaint_terms":list(f[o[::-1][:18]]),"praise_terms":list(f[o[:18]])}
print("text AUC",R["text_model"]["auc"],"\ncomplaints:",R["text_model"]["complaint_terms"],"\npraise:",R["text_model"]["praise_terms"])
V1=V[(V.y<=2022)&(V.y>=2013)]; sea={}
for t,g in V1.groupby("type"):
    if len(g)>=1500:
        s=g.groupby("m").size().reindex(range(1,13),fill_value=0); sea[t]=(s/s.mean()).round(2).tolist()
sea["All"]=(lambda s:(s/s.mean()).round(2).tolist())(V1.groupby("m").size().reindex(range(1,13),fill_value=0)); R["seasonality"]=sea
print({k:v for k,v in sea.items() if k in("All","Notebook","Highlighter")})
R["review_years"]=V.y.value_counts().sort_index().to_dict()

# ---------- 5. predictive model: will a product be rated below 4.2? ----------
F=pd.DataFrame({"lp":np.log(M.price),"ln":np.log(M.rating_number),"pk":np.log(M.pack),"bn":np.log(M.groupby("brand").brand.transform("size")),
  "feat":M.features.fillna("").str.len(),"tlen":M.title.str.len(),"type":M.type.astype("category").cat.codes})
y=(M.average_rating<4.2).astype(int); R["model"]={"base_rate":round(y.mean(),3)}
clf=HistGradientBoostingClassifier(max_depth=4,learning_rate=.06,max_iter=200,categorical_features=[6],random_state=0)
cvs=cross_val_score(clf,F,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),scoring="roc_auc"); R["model"]["auc"]=round(cvs.mean(),3); R["model"]["auc_sd"]=round(cvs.std(),3)
from sklearn.inspection import permutation_importance
clf.fit(F,y); pi=permutation_importance(clf,F.sample(5000,random_state=0),y.loc[F.sample(5000,random_state=0).index],scoring="roc_auc",n_repeats=3,random_state=0)
nm={"lp":"price","ln":"number of ratings","pk":"pack size","bn":"brand catalogue size","feat":"feature-text length","tlen":"title length","type":"product type"}
R["model"]["importance"]={nm[c]:round(v,3) for c,v in sorted(zip(F.columns,pi.importances_mean),key=lambda z:-z[1])}
print("model",R["model"])

# ---------- 6. hidden gems + export ----------
G=[]
for t,g in M.groupby("type"):
    if len(g)<150: continue
    q=g[(g.rating_number.between(25,400))&(g.price<=g.price.median())&(g.average_rating>=4.6)].sort_values("bayes",ascending=False).head(4)
    G+=[[t,r.title[:70],r.brand,round(r.price,2),r.average_rating,int(r.rating_number)] for r in q.itertuples()]
R["gems"]=G
S=M[M.type.map(M.type.value_counts())>=150]
S=S.sample(frac=1,random_state=1).groupby("type").head(220)
R["sample"]=[[a,b_,round(c,2),d,int(e),f_[:60]] for a,b_,c,d,e,f_ in zip(S.type,S.brand,S.price,S.average_rating,S.rating_number,S.title)]
R["types"]={t:{"n":int(len(g)),"median_price":round(g.price.median(),2),"mean_rating":round(g.average_rating.mean(),2),"ratings":int(g.rating_number.sum()),"median_unit":round(g.unit_price.median(),2)} for t,g in M.groupby("type")}
json.dump(R,open("data/results.json","w"),default=float); M.to_csv("data/clean_products.csv",index=False)
import os; print("results.json KB:",os.path.getsize("data/results.json")//1024)
