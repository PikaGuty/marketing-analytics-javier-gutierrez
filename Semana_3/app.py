"""Aplicación completada a partir de material_original/streamlit_starter.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
from analisis import ejecutar, figura, politica

st.set_page_config(page_title='Grupo 1 · Uplift y Churn', page_icon='☕', layout='wide')
st.title('Starbucks Rewards: ¿a quién contactar?')
st.caption('Laboratorio educativo con datos simulados: 40.000 clientes, cupón de $5 y horizonte de 60 días.')

@st.cache_data
def resultados():
    return ejecutar()

r = resultados()
optimo = int(r['tabla'].loc[r['tabla'].valor_total.idxmax(),'pct_contactado'])
with st.sidebar:
    st.header('Política de contacto')
    pct = st.slider('Porcentaje de clientes a contactar', 10,100,optimo,10)
    nombre = st.selectbox('Modelo dentro del T-learner',r['comparacion'].modelo.tolist())
    st.caption('Partición fija: 70% entrenamiento, 30% evaluación; semilla 42. La política principal del laboratorio usa logística.')
p = politica(r['test'],'uplift_'+nombre,pct)
cols=st.columns(3)
cols[0].metric('Clientes de evaluación a contactar',f"{p['n_contactados']:,}")
cols[1].metric('Valor incremental neto por contactado',f"${p['valor_por_cliente']:+.2f}")
cols[2].metric('Valor incremental total en evaluación',f"${p['valor_total']:+,.2f}")
st.pyplot(figura(r))
st.subheader('Políticas del Paso 3 · T-learner logístico')
st.dataframe(r['tabla'].round(2),hide_index=True)
st.write(f'El máximo observado en la cuadrícula es {optimo}%. Es una recomendación exploratoria: elegir el porcentaje en estos mismos datos introduce optimismo y requiere una nueva prueba antes de implementarlo.')
st.subheader('Comparación de modelos en los mismos 12.000 clientes')
st.dataframe(r['comparacion'].round(4),hide_index=True)
st.markdown('**Correlación:** Pearson entre uplift estimado y probabilidad incremental verdadera, revelada después de entrenar. **Qini:** área de la curva menos la referencia al azar, dividida por N, según el notebook original; mide respuestas, no rentabilidad.')
st.caption('La curva original calcula (tasa tratada − tasa control) × N en cada prefijo. No es un conteo acumulado estándar; el primer prefijo puede contener sólo un brazo y resultar inestable. Se conserva para comparar según el laboratorio.')
st.info('Valor económico: media de utilidad_neta_60d entre tratados seleccionados menos media de margin_60d entre controles seleccionados. Cupón: $5 si compra; envío: $0,05 por tratado. Los importes totales corresponden a la muestra de evaluación, no a toda la empresa.')
st.download_button('Descargar comparación CSV',r['comparacion'].to_csv(index=False).encode('utf-8'),file_name='comparacion_modelos.csv',mime='text/csv')
pdf = Path(__file__).resolve().parent/'conclusion.pdf'
if pdf.exists():
    st.download_button('Descargar conclusión para Marketing',pdf.read_bytes(),file_name='conclusion.pdf',mime='application/pdf')
