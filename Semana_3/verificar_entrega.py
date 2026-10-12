"""Verificación de artefactos e interacción real de Streamlit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nbformat
from streamlit.testing.v1 import AppTest
from analisis import BASE

nb=nbformat.read(BASE/'laboratorio_final.ipynb',as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
assert not any(o.output_type=='error' for c in code for o in c.outputs)
assert 'revelar_verdad=False' in code[0].source
for nombre in ['conclusion.pdf','comparacion_modelos.csv','tabla_politicas.csv','curva_qini_laboratorio.png']:
    assert (BASE/nombre).stat().st_size>0
app=AppTest.from_file(str(BASE/'app.py'),default_timeout=120).run()
assert not app.exception, app.exception
pd.testing.assert_frame_equal(app.dataframe[0].value,pd.read_csv(BASE/'tabla_politicas.csv').round(2),check_dtype=False)
pd.testing.assert_frame_equal(app.dataframe[1].value,pd.read_csv(BASE/'comparacion_modelos.csv').round(4),check_dtype=False)
assert app.slider[0].value==20
before=app.metric[0].value
app.slider[0].set_value(30).run()
assert not app.exception and app.metric[0].value != before
app.selectbox[0].select('Random Forest').run()
assert not app.exception
print('OK: notebook completo, sin errores; CSV idénticos a la app; slider y selector funcionan.')
(BASE/'verificacion.json').write_text(json.dumps({'notebook_celdas_ejecutadas':len(code),'errores':0,'csv_app_coinciden':True,'slider_verificado':True,'selector_verificado':True},indent=2),encoding='utf-8')
