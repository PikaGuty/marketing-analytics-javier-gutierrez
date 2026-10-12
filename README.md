# Actividad 2 · Uplift y Churn · Grupo 1

Entrega en Semana_3, rama `Tarea_Actividad_2_Grupo_1`. Campaña educativa simulada de Starbucks Rewards: 40.000 clientes, cupón de $5 y horizonte de 60 días. No son resultados comerciales reales de Starbucks.

## Instalar y ejecutar

Desde la raíz, con Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Semana_3/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run Semana_3/app.py
```

La aplicación permite elegir porcentaje de contacto y modelo. Muestra Qini, tabla de políticas y comparación; comparte las funciones del notebook en `Semana_3/analisis.py`. Puede iniciarse desde cualquier directorio.

## Reproducir los entregables

```powershell
.\.venv\Scripts\python.exe Semana_3/preparar_entrega.py
```

Este comando entrena, calcula métricas, genera CSV y conclusión PDF y ejecuta el notebook de principio a fin conservando sus salidas. También se puede abrir `Semana_3/laboratorio_final.ipynb` en Jupyter o VS Code y ejecutar todas las celdas desde la raíz o Semana_3.

Entregables: laboratorio_final.ipynb, comparacion_modelos.csv, tabla_politicas.csv, curva_qini_laboratorio.png, conclusion.pdf y conclusion.md.

## Diseño y métricas

Partición original 70/30, estratificada por tratamiento, semilla 42. Los cuatro T-learners se evalúan sobre los mismos 12.000 clientes. La logística usa escalado aprendido sólo dentro de cada brazo de entrenamiento. La verdad y los resultados económicos no se usan como predictores.

Se comparan logística, árbol sin límite, árbol con profundidad 2 y Random Forest regularizado. La correlación usa el uplift verdadero, revelado tras el ajuste. Qini conserva la fórmula particular del notebook original: diferencia de tasas por prefijo multiplicada por N, integración trapezoidal, resta de la referencia discretizada y división por N. No equivale al conteo incremental acumulado estándar; se explica en notebook y aplicación.

Las políticas comparan utilidad neta de tratados con margen de controles del segmento. Cupón: $5 si hay compra; envío: $0,05 por tratado. La tabla principal sigue la logística. Buscar el mejor porcentaje y comparar modelos en la misma evaluación genera optimismo: se requiere un nuevo experimento confirmatorio antes de aplicar la recomendación. Los totales corresponden a la evaluación, no a toda la base.

## Procedencia

El PDF proporcionado identifica el [material del profesor](https://github.com/stanlybolanos/marketing-analytics-base-2026/tree/main/Semana_3/Tarea). Notebook base, starter, requisitos y dataset completo están preservados en `Semana_3/material_original`; el Laboratorio.ipynb vacío existente se conserva intacto. El dataset sin verdad está en Semana_3/data/uplift_campaign.csv. fuentes.json registra hashes SHA-256.

## Desplegar en Streamlit Community Cloud

1. Iniciar sesión con una cuenta con acceso al repositorio.
2. Crear una aplicación con repositorio `PikaGuty/marketing-analytics-javier-gutierrez`, rama `Tarea_Actividad_2_Grupo_1` y archivo principal `Semana_3/app.py`.
3. Seleccionar Python 3.11 en opciones avanzadas. La carpeta de la app contiene requirements.txt.
4. Desplegar y verificar curva, políticas y controles. Registrar la URL real.

No se registra una URL de aplicación hasta verificar que la publicación existe.

## Resultados y verificaciones de esta entrega

| T-learner | Correlación uplift real | Qini |
|---|---:|---:|
| Logística | 0,618635 | 0,141527 |
| Árbol sin límite | 0,203512 | 0,065821 |
| Árbol profundidad 2 | 0,605265 | 0,126014 |
| Random Forest | 0,862467 | 0,157588 |

Política logística: 20% maximiza el valor total observado ($2.275,78 sobre 2.400 contactos de evaluación); 10% maximiza el valor por persona. La comparación de modelos es exploratoria y Random Forest mejora ambas métricas frente a logística.

Verificado: notebook con 7 celdas de código ejecutadas sin errores; conclusión PDF de una página; tablas de la app idénticas a CSV; slider y selector funcionales; `pip check` sin incompatibilidades; inicio de Streamlit y salud HTTP 200.

Para repetir las verificaciones de notebook y app:

```powershell
.\.venv\Scripts\python.exe Semana_3/verificar_entrega.py
```
