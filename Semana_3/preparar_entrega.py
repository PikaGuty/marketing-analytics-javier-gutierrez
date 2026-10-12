"""Genera y ejecuta los entregables a partir del material oficial."""
from pathlib import Path
import json
import hashlib
import sys
import nbformat as nbf
from nbclient import NotebookClient
from analisis import BASE, ejecutar
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

def generar():
    r = ejecutar()
    tabla, comp = r['tabla'], r['comparacion']
    mejor = tabla.loc[tabla.valor_total.idxmax()]
    unitario = tabla.loc[tabla.valor_por_cliente.idxmax()]
    corr = comp.set_index('modelo').loc['Logística','correlacion_uplift_real']
    qini = comp.set_index('modelo').loc['Logística','indice_qini']
    ganador = comp.loc[comp.indice_qini.idxmax()]
    verdad = r['verdad']
    elegidos = verdad.sort_values('uplift_estimado',ascending=False).head(int(mejor.n_contactados))
    oracle = verdad.sort_values('true_value_of_contact',ascending=False).head(int(mejor.n_contactados))
    valor_real, valor_oracle = elegidos.true_value_of_contact.sum(), oracle.true_value_of_contact.sum()
    r['comparacion'].to_csv(BASE/'comparacion_modelos.csv',index=False)
    tabla.to_csv(BASE/'tabla_politicas.csv',index=False)
    parrafos = [
        'Señor Presidente de Marketing: recomendamos evaluar en un nuevo experimento una política de contacto dirigida al '+f'{int(mejor.pct_contactado)}% de la base, ordenada por el uplift del T-learner logístico. Esta recomendación proviene de la campaña educativa de Starbucks Rewards: 40.000 clientes, cupón de $5 y medición a 60 días. Los datos son simulados, con tipos y efectos verdaderos disponibles para auditoría; no representan evidencia comercial real de Starbucks.',
        f'En los 12.000 clientes reservados para evaluación, esa política contacta a {int(mejor.n_contactados):,} personas y produce una estimación incremental neta de ${mejor.valor_por_cliente:+.2f} por contactado, equivalente a ${mejor.valor_total:+,.2f} en esta muestra. Es el máximo de los diez porcentajes examinados. El mayor valor unitario aparece al {int(unitario.pct_contactado)}%: priorizar rentabilidad por persona y maximizar valor total son objetivos distintos. Contactar al 30% de mayor riesgo de churn estima ${r["riesgo30"]["valor_total"]:+,.2f}; predecir quién no compra no identifica necesariamente a quien cambia su comportamiento por el cupón.',
        f'El T-learner logístico obtiene Qini {qini:+.4f}, frente a {r["qini_riesgo"]:+.4f} del ranking por riesgo. Tras revelar la verdad, la correlación de Pearson entre uplift estimado y verdadero es {corr:.4f}. En la comparación de cuatro clasificadores, {ganador.modelo} alcanza el mayor Qini ({ganador.indice_qini:+.4f}); la tabla adjunta permite contrastar también la correlación. Qini evalúa ordenamiento de respuestas, no ganancia económica, y su fórmula aquí es la específica del laboratorio.',
        f'La auditoría final aporta una referencia adicional: los contactos elegidos por logística suman ${valor_real:+,.2f} de valor esperado verdadero, frente a ${valor_oracle:+,.2f} si pudiéramos elegir con conocimiento perfecto, para igual número de contactos. Esta diferencia cuantifica lo que deja de capturar el modelo. La verdad se usó sólo después de entrenar y evaluar; nunca como predictor ni para elegir el porcentaje.',
        'La recomendación supone asignación aleatoria, comparabilidad de tratados y controles, ausencia de interferencia entre clientes y atributos medidos antes de la campaña. La utilidad descuenta $5 sólo cuando hay compra y $0,05 por envío. No incluye efectos posteriores a 60 días, capacidad operativa ni costos adicionales. Se entrenó en el 70% de la base con semilla fija; la búsqueda del porcentaje sobre el mismo 30% de evaluación puede sobreestimar su beneficio, y la comparación de modelos también es exploratoria. No presentamos estos máximos como una prueba independiente de desempeño.',
        'Antes de extender la política, recomendamos una prueba aleatoria nueva en el porcentaje propuesto, con un control sin cupón y un criterio de utilidad neta acordado previamente. La extrapolación a toda la base exige estabilidad de clientes, costos y efectos; los importes reportados corresponden únicamente a la evaluación. El avance ejecutivo consiste en priorizar valor incremental, validar la decisión y evitar gastar en clientes que comprarían de todos modos.'
    ]
    conclusion = '# Conclusión dirigida al Presidente de Marketing\n\n'+'\n\n'.join(parrafos)
    (BASE/'conclusion.md').write_text(conclusion,encoding='utf-8')
    styles=getSampleStyleSheet()
    styles['BodyText'].fontSize=10; styles['BodyText'].leading=13
    doc=SimpleDocTemplate(str(BASE/'conclusion.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=35)
    flow=[Paragraph('Recomendación de contacto — Grupo 1',styles['Title']),Spacer(1,10)]
    for p in parrafos: flow += [Paragraph(p,styles['BodyText']),Spacer(1,8)]
    doc.build(flow)
    original=nbf.read(BASE/'material_original/laboratorio_final.ipynb',as_version=4)
    cells=[]
    def md(t): cells.append(nbf.v4.new_markdown_cell(t))
    def code(t): cells.append(nbf.v4.new_code_cell(t))
    md(original.cells[0].source+'\n\n**Grupo 1.** Fuente: material oficial de Semana_3/Tarea. Datos educativos simulados. La verdad oculta se consulta sólo en la revelación final. La partición y funciones originales se conservan; el escalado se aprende dentro de cada brazo con entrenamiento exclusivamente.')
    code("from pathlib import Path\nimport sys\nimport pandas as pd\nimport numpy as np\nfrom IPython.display import display\nBASE = Path.cwd() if (Path.cwd()/'analisis.py').exists() else Path.cwd()/'Semana_3'\nsys.path.insert(0,str(BASE))\nfrom analisis import ejecutar, revelar, figura, politica, FEATURES, SEED\nr = ejecutar(revelar_verdad=False)\ntrain, test = r['train'], r['test']\nprint(f'{len(train)+len(test):,} clientes; entrenamiento {len(train):,}; evaluación {len(test):,}; semilla {SEED}')\ndisplay(test.head())")
    md(original.cells[2].source)
    code("print('Predictores anteriores al tratamiento:', FEATURES)\nprint('AUC churn:', r['auc_riesgo'])\nprint('Política de riesgo al 30%:',r['riesgo30'])\nassert set(train.customer_id).isdisjoint(test.customer_id)")
    md('**Respuesta Paso 1.** El churn se define como 1 − responded_60d. La logística aprende en entrenamiento, sin variables de resultados, tratamiento o verdad como predictores. Su AUC mide capacidad predictiva, no causalidad. El valor de la política se estima comparando utilidad neta de tratados y margen de controles del mismo segmento; el resultado numérico está arriba. El riesgo combina respuestas de ambos brazos según el enunciado, por lo que no equivale al riesgo sin campaña.')
    md(original.cells[6].source)
    code("import inspect\nimport analisis\nprint(inspect.getsource(analisis.modelos))\nprint(inspect.getsource(analisis.ejecutar))\nprint('Qini logística:',r['comparacion'].set_index('modelo').loc['Logística','indice_qini'])\nprint('Qini riesgo:',r['qini_riesgo'])")
    md('**Respuesta Paso 2.** Cada T-learner estima p(Y=1|X,T=1) y p(Y=1|X,T=0) en brazos separados y resta ambas probabilidades sobre la misma evaluación. El StandardScaler de cada logística se ajusta exclusivamente en su brazo de entrenamiento. Los cuatro clasificadores y sus hiperparámetros se fijaron antes de consultar la verdad. No se utilizan identificadores, margen, utilidad ni respuesta como atributos.')
    code("print(inspect.getsource(analisis.curva_qini))\nprint(inspect.getsource(analisis.coeficiente_qini))\nfig=figura(r)\nfig.savefig(BASE/'curva_qini_laboratorio.png',dpi=150)\ndisplay(fig)")
    md('**Definición de Qini conservada.** En cada prefijo se calcula la tasa de respuesta tratada menos la tasa control y se multiplica por N, el tamaño completo de evaluación. Se muestrean 40 posiciones más el origen. El índice es [área trapezoidal de la curva − área trapezoidal de np.linspace(0, y_final, len(xs)) sobre los mismos xs] / N. Se mantiene incluso la referencia discretizada del notebook. Esta curva no es el conteo acumulado estándar: usa N, no el tamaño del prefijo. El primer prefijo puede contener un solo brazo y crea inestabilidad; con árboles también hay empates. Un Qini mayor señala mejor ranking bajo esta convención, sin implicar mayor utilidad monetaria.')
    md(original.cells[10].source)
    code("print(inspect.getsource(analisis.valor_de_la_politica))\ndisplay(r['tabla'].round(2))\nr['tabla'].to_csv(BASE/'tabla_politicas.csv',index=False)\nmejor=r['tabla'].loc[r['tabla'].valor_total.idxmax()]\nunitario=r['tabla'].loc[r['tabla'].valor_por_cliente.idxmax()]\nprint('Óptimo total:',mejor.to_dict())\nprint('Óptimo unitario:',unitario.to_dict())")
    md(original.cells[12].source+f'\n\n**Respuesta.** El máximo total corresponde al {int(mejor.pct_contactado)}%, con ${mejor.valor_total:+,.2f}; el máximo por contactado corresponde al {int(unitario.pct_contactado)}%. No necesariamente coinciden: el valor total multiplica el valor unitario por el volumen. Se mantienen los valores sin redondear para elegir el máximo. Aunque el texto menciona value_60d, la función original y el starter utilizan utilidad_neta_60d frente a margin_60d: seguimos esa definición económica. La selección entre diez porcentajes es exploratoria y su máximo necesita evaluación independiente. Los totales no representan una ganancia ya realizada ni toda la base de 40.000.')
    md('## Paso 4 — Auditoría del diseño\n\nLa muestra de evaluación es idéntica para los cuatro modelos (12.000 clientes); entrenamiento tiene 28.000. No hay fuga hacia el ajuste de modelos: se utiliza una lista cerrada de cinco atributos previos y escalado dentro de cada brazo. La búsqueda de política y comparación sobre evaluación sí generan sesgo de selección; no se reutiliza esta muestra como evidencia confirmatoria de la política seleccionada. Se exige una nueva prueba. Los datos son simulados con verdad conocida, no resultados comerciales verificables.')
    code("print(train.treatment_group.value_counts())\nprint(test.treatment_group.value_counts())\nassert not any(c.startswith('true_') for c in FEATURES)\nassert test.customer_id.is_unique\nassert np.isfinite(r['comparacion'][['indice_qini']]).all().all()")
    md('## Paso 5 — La revelación final y comparación de modelos\n\n'+original.cells[13].source+'\n\nLa verdad se une por customer_id con validación uno a uno. La correlación es Pearson contra true_uplift_prob, nunca contra la respuesta observada. Se conserva la logística como política principal del Paso 3; los otros modelos son una comparación exploratoria.')
    code("revelar(r)\nverdad=r['verdad']\ndisplay((verdad.true_type.value_counts(normalize=True)*100).round(2))\ndisplay(r['comparacion'])\nr['comparacion'].to_csv(BASE/'comparacion_modelos.csv',index=False)\nseleccion=verdad.sort_values('uplift_estimado',ascending=False).head(int(mejor.n_contactados))\noracle=verdad.sort_values('true_value_of_contact',ascending=False).head(int(mejor.n_contactados))\nprint('Valor esperado verdadero de la política:',seleccion.true_value_of_contact.sum())\nprint('Valor esperado con conocimiento perfecto, mismo volumen:',oracle.true_value_of_contact.sum())\nprint('Composición de contactos elegidos:')\ndisplay((seleccion.true_type.value_counts(normalize=True)*100).round(2))")
    md(f'**Respuesta comparación.** La correlación logística es {corr:.4f}. El mayor Qini corresponde a {ganador.modelo}, con {ganador.indice_qini:+.4f}. La tabla muestra qué modelo mejora cada criterio; correlación y Qini no evalúan exactamente lo mismo y no implican calibración perfecta ni beneficio económico. El árbol sin límite puede memorizar entrenamiento y producir diferencias extremas de probabilidades; profundidad y tamaño mínimo de hoja regularizan el Random Forest. La auditoría de valor compara efectos esperados simulados, mientras el Paso 3 compara resultados observados aleatorios: son estimandos relacionados pero no números idénticos.')
    md(conclusion)
    md('## Reproducibilidad y entregables\n\nEjecutar desde la raíz o Semana_3 con las dependencias de Semana_3/requirements.txt. analisis.py comparte exactamente las funciones con Streamlit. Se exportan comparacion_modelos.csv, tabla_politicas.csv, curva_qini_laboratorio.png y conclusion.pdf. Fuente original preservada en material_original. No se sustituyeron los datos ni se copiaron las métricas de ejemplo del enunciado.')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    cliente=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(BASE)}})
    cliente.execute()
    nbf.write(nb,BASE/'laboratorio_final.ipynb')
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'data/uplift_campaign.csv',BASE/'material_original/uplift_campaign_FULL.csv']}
    (BASE/'fuentes.json').write_text(json.dumps({'origen':'https://github.com/stanlybolanos/marketing-analytics-base-2026/tree/main/Semana_3/Tarea','sha256':hashes,'seed':42,'test_size':.3},indent=2),encoding='utf-8')
    print(comp.to_string(index=False));print(tabla.to_string(index=False)); print('PDF y notebook ejecutado guardados')

if __name__=='__main__': generar()
