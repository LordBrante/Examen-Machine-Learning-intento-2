# Clasificación sobre Bank Marketing Dataset

Notebook de examen que aborda un problema de clasificación binaria sobre el **Bank Marketing Dataset** de UCI: predecir si un cliente de una institución bancaria portuguesa suscribirá o no un depósito a plazo, a partir de datos de campañas de telemarketing.

## Contenido

El notebook está organizado en 8 secciones secuenciales:

| # | Sección | Descripción |
|---|---------|-------------|
| 0 | Configuración del entorno | Montaje de Google Drive, librerías y funciones reutilizables |
| 1 | Carga y descripción del dataset | Comparación de los 4 archivos originales de UCI y selección/combinación de `bank.csv` + `bank-full.csv` |
| 2 | Limpieza de Datos | Tipos de datos, duplicados, valores `unknown` → `NaN`, imputación, outliers (IQR) y análisis de **data leakage** (`duration`) |
| 3 | Exploración de Datos (EDA) | Estadísticas descriptivas, visualizaciones univariadas y multivariadas, distribución del target, mapa de correlación |
| 4 | Preparación del dataset | Definición de features/target, train/test split estratificado, `ColumnTransformer` (imputación + escalado/codificación) |
| 5 | Modelos base | Decision Tree y SVM (lineal) con pipelines de scikit-learn, validación cruzada estratificada |
| 6 | Optimización de hiperparámetros | `GridSearchCV` para Decision Tree y SVM |
| 7 | Evaluación y comparación | Tabla de métricas, matrices de confusión, gráfico comparativo, curvas ROC/AUC, importancia de features |
| 8 | Conclusiones | Interpretación del desbalance de clases y selección del mejor modelo |

## Dataset

- **Fuente:** [Bank Marketing Dataset – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Archivos usados:** `bank.csv` + `bank-full.csv` (se descartan las versiones `additional`, ver Sección 1.2 del notebook para la justificación)
- **Target:** `y` — si el cliente suscribió (`1`) o no (`0`) un depósito a plazo
- **Desbalance:** ~88% No / ~12% Sí
- **Nota importante:** la variable `duration` se excluye del modelado por **data leakage** temporal (no se conoce antes de realizar la llamada)

## Requisitos

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

Librerías principales utilizadas:
- `pandas`, `numpy` — manipulación de datos
- `matplotlib`, `seaborn` — visualización
- `scipy.stats` — análisis estadístico
- `scikit-learn` — pipelines, preprocesamiento, modelado (`DecisionTreeClassifier`, `LinearSVC`/`SVC`), `GridSearchCV`, métricas

## Cómo ejecutar

Este notebook fue desarrollado en **Google Colab** y espera los archivos del dataset en:

```
/content/drive/MyDrive/Colab Notebooks/datasets/Bank/
```

con los archivos `bank.csv`, `bank-full.csv` y sus versiones `additional` (separados por `;`).

Pasos:
1. Subir los 4 archivos CSV del dataset a la ruta de Drive indicada (o ajustar la variable `origen` en la Sección 0 si se usa otra ubicación).
2. Ejecutar las celdas en orden de arriba hacia abajo — cada sección depende de las anteriores (limpieza → EDA → preparación → modelado → evaluación).
3. Si se ejecuta fuera de Colab, eliminar/adaptar la celda de `drive.mount(...)` y apuntar `origen` a una ruta local.

## Modelos evaluados

- **Decision Tree Classifier** (base y optimizado vía `GridSearchCV`)
- **SVM lineal** (`LinearSVC`, base y optimizado vía `GridSearchCV`)

Métricas reportadas: Accuracy, Precision, Recall, F1-Score, AUC-ROC. Dado el fuerte desbalance de clases, el notebook prioriza **F1-Score** y **Recall de la clase positiva** sobre Accuracy para comparar modelos.

## Resultado

La conclusión del análisis (Sección 8.3) favorece al **Decision Tree Optimizado** por su interpretabilidad y menor costo computacional en un contexto bancario, aunque el **SVM Optimizado** suele obtener mejor F1-Score y AUC. Ver el notebook para el detalle completo del trade-off.
