"""Lógica compartida; definiciones económicas y Qini del notebook oficial."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent
FEATURES = ['recency_days', 'frequency', 'monetary', 'email_open_rate', 'tenure_days']
SEED = 42

def modelos():
    return {
        'Logística': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        'Árbol sin límite': DecisionTreeClassifier(random_state=SEED),
        'Árbol max_depth=2': DecisionTreeClassifier(max_depth=2, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=50, random_state=SEED, n_jobs=2),
    }

def curva_qini(test_df, score_col, steps=40):
    # Conserva exactamente la definición y muestreo del material original.
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    tratado = (orden.treatment_group == 'treatment').values
    y = orden.responded_60d.values
    rt, rc = np.cumsum(np.where(tratado, y, 0)), np.cumsum(np.where(~tratado, y, 0))
    nt, nc = np.cumsum(tratado), np.cumsum(~tratado)
    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        t = rt[i-1] / nt[i-1] if nt[i-1] else 0
        c = rc[i-1] / nc[i-1] if nc[i-1] else 0
        xs.append(i/n)
        ys.append((t-c)*n)
    return np.array(xs), np.array(ys)

def coeficiente_qini(xs, ys, n):
    return (np.trapezoid(ys, xs) - np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs))/n

def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    vt = sel.loc[sel.treatment_group == 'treatment', 'utilidad_neta_60d'].mean()
    vc = sel.loc[sel.treatment_group == 'control', 'margin_60d'].mean()
    return vt-vc, len(sel)

def politica(test, score, pct):
    idx = test.sort_values(score, ascending=False).head(int(len(test)*pct/100)).index
    v, n = valor_de_la_politica(test, idx)
    return dict(pct_contactado=pct, valor_por_cliente=v, n_contactados=n, valor_total=v*n)

def ejecutar(revelar_verdad=True):
    df = pd.read_csv(BASE/'data/uplift_campaign.csv')
    assert len(df) == 40000 and df.customer_id.is_unique
    assert not df[FEATURES].isna().any().any()
    train, test = train_test_split(df, test_size=.30, random_state=SEED, stratify=df.treatment_group)
    assert set(train.customer_id).isdisjoint(test.customer_id)
    test = test.copy()
    riesgo = clone(modelos()['Logística']).fit(train[FEATURES], 1-train.responded_60d)
    test['prob_churn'] = riesgo.predict_proba(test[FEATURES])[:,1]
    filas, curvas = [], {}
    for nombre, modelo in modelos().items():
        t, c = train[train.treatment_group=='treatment'], train[train.treatment_group=='control']
        mt = clone(modelo).fit(t[FEATURES], t.responded_60d)
        mc = clone(modelo).fit(c[FEATURES], c.responded_60d)
        score = 'uplift_'+nombre
        test[score] = mt.predict_proba(test[FEATURES])[:,1]-mc.predict_proba(test[FEATURES])[:,1]
        xs, ys = curva_qini(test, score)
        curvas[nombre] = (xs, ys)
        filas.append(dict(modelo=nombre, indice_qini=coeficiente_qini(xs,ys,len(test))))
    test['uplift_estimado'] = test['uplift_Logística']
    tabla = pd.DataFrame([politica(test,'uplift_estimado',p) for p in range(10,101,10)])
    comparacion = pd.DataFrame(filas)
    xr, yr = curva_qini(test,'prob_churn')
    resultado = dict(train=train,test=test,comparacion=comparacion,tabla=tabla,curvas=curvas,
                curva_riesgo=(xr,yr),qini_riesgo=coeficiente_qini(xr,yr,len(test)),
                auc_riesgo=roc_auc_score(1-test.responded_60d,test.prob_churn),riesgo30=politica(test,'prob_churn',30))
    if revelar_verdad:
        revelar(resultado)
    return resultado

def revelar(r):
    """Auditoría posterior; no modifica modelos, scores ni políticas."""
    df = pd.read_csv(BASE/'data/uplift_campaign.csv')
    full = pd.read_csv(BASE/'material_original/uplift_campaign_FULL.csv')
    pd.testing.assert_frame_equal(df, full[df.columns])
    verdad = r['test'].merge(full[['customer_id','true_type','true_uplift_prob','true_value_of_contact']], on='customer_id', validate='one_to_one')
    r['verdad'] = verdad
    r['comparacion']['correlacion_uplift_real'] = [verdad['uplift_'+m].corr(verdad.true_uplift_prob) for m in r['comparacion'].modelo]
    r['comparacion'] = r['comparacion'][['modelo','correlacion_uplift_real','indice_qini']]

def figura(r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    for nombre, (x,y) in r['curvas'].items():
        q = r['comparacion'].set_index('modelo').loc[nombre,'indice_qini']
        ax.plot(x,y,label=f'{nombre} (Qini={q:+.3f})')
    x,y = r['curva_riesgo']
    ax.plot(x,y,label=f"Riesgo (Qini={r['qini_riesgo']:+.3f})",color='brown')
    ax.plot([0,1],[0,y[-1]],'--',color='gray',label='Referencia al azar')
    ax.set(xlabel='Fracción contactada de la base de prueba',ylabel='Diferencia de tasas × N',title='Qini según la definición del laboratorio')
    ax.legend(); fig.tight_layout()
    return fig
