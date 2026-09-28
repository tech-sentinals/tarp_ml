import pickle, json, numpy as np, pandas as pd
b = pickle.load(open("outputs/selected_model.pkl","rb"))
pipe, tau = b["pipeline"], b["protective_tau"]
pre, gb = pipe.named_steps["pre"], pipe.named_steps["clf"]
NUM = list(pre.transformers_[0][2]); cats = list(pre.named_transformers_["cat"].categories_[0])
df = pd.read_csv("data/intervention_records.csv")
X = df[NUM + ["actuator_type"]]
Xt = pre.transform(X[:50]); Xt = Xt.toarray() if hasattr(Xt,"toarray") else Xt
init = gb._raw_predict_init(Xt[:1])[0].tolist()
trees = []
for stage in gb.estimators_:
    st = []
    for t in stage:
        tr = t.tree_
        st.append([tr.feature.tolist(), [float(v) for v in tr.threshold.tolist()],
                   tr.children_left.tolist(), tr.children_right.tolist(),
                   [round(v,7) for v in tr.value[:,0,0].tolist()]])
    trees.append(st)
m = {"num": NUM, "cats": cats, "classes": list(gb.classes_), "lr": gb.learning_rate,
     "init": init, "tau": tau, "trees": trees}
# verify
def raw(x):
    r = np.array(init, dtype=float)
    for stage in trees:
        for k,(f,th,l,rr,v) in enumerate(stage):
            n=0
            while l[n]!=-1:
                n = l[n] if x[f[n]] <= th[n] else rr[n]
            r[k]+= gb.learning_rate*v[n]
    e=np.exp(r-r.max()); return e/e.sum()
mine = np.array([raw(x) for x in Xt]); ref = gb.predict_proba(Xt)
print("max abs diff", np.abs(mine-ref).max())
js = "const MODEL = " + json.dumps(m, separators=(",", ":")) + ";\n"
open("app/model_export.js", "w").write(js)
print("wrote app/model_export.js,", len(js) // 1024, "KB")
