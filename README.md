# Proyecto 1 — Prediccion de Tarifas de Transporte en NYC

**Curso:** Inteligencia Artificial CC3085 — Universidad del Valle de Guatemala  
**Catedratico:** Alan Reyes  
**Seccion:** 20  
**Integrantes:**
- Joel Jaquez (23369)
- Diego Patzan (23525)
- Luis Gonzalez (23353)

---

## Descripcion General

Desarrollo de un agente reactivo basado en modelos de machine learning supervisado para regresion. El objetivo es construir un modelo predictivo capaz de estimar el costo de un viaje en taxi en la ciudad de Nueva York, dadas las coordenadas geograficas de origen y destino, la fecha, la hora y el numero de pasajeros.

El dataset utilizado proviene de la competencia publica de Kaggle [New York City Taxi Fare Prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data). Dado el volumen original (~55 millones de registros), se trabajo con una muestra representativa de **1,000,000 de filas**.

---

## Estructura del Repositorio

```
Proyecto1-IA/
├── data/
│   └── train.csv                        # Dataset original (no incluido en el repo)
├── notebook/
│   ├── proyecto1.ipynb                  # Notebook principal con todo el pipeline
│   ├── Limpieza y carga/
│   │   ├── distribucion_antes_limpieza.png
│   │   ├── distribucion_despues_limpieza.png
│   │   ├── correlacion.png
│   │   └── feature_engineering.png
│   ├── Regresion Lineal Multivariada/
│   │   ├── lr_predicciones_vs_reales.png
│   │   └── lr_residuos.png
│   ├── Random Forest/
│   │   ├── rf_predicciones_vs_reales.png
│   │   ├── rf_residuos.png
│   │   └── rf_importancia_features.png
│   └── Redes Neuronales Densas/
│       ├── mlp_curvas_aprendizaje.png
│       ├── mlp_predicciones_vs_reales.png
│       └── mlp_residuos.png
├── reporte/
│   └── Proyecto1_IA.pdf                 # Reporte tecnico completo
├── Instrucciones/
│   └── Proyecto1.pdf
├── .venv312/                            # Entorno virtual Python 3.12 (ignorado por git)
├── .gitignore
└── README.md
```

---

## Entorno y Dependencias

**Python:** 3.12.5 (requerido — TensorFlow no es compatible con Python 3.13+)

**Crear entorno virtual e instalar dependencias:**

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter ipykernel
```

**Registrar el kernel en Jupyter/VS Code:**

```bash
python -m ipykernel install --user --name=venv312 --display-name "Python 3.12 (proyecto)"
```

**Bibliotecas principales:**

| Biblioteca | Version | Uso |
|---|---|---|
| pandas | 3.0.2 | Manipulacion y limpieza de datos |
| numpy | 2.4.4 | Operaciones numericas y formula Haversine |
| matplotlib | 3.10.8 | Visualizaciones y graficas |
| seaborn | 0.13.2 | Mapa de calor de correlaciones |
| scikit-learn | 1.8.0 | Regresion Lineal, Random Forest, metricas, split |
| tensorflow | 2.21.0 | Red Neuronal Densa (MLP) |
| keras | 3.14.0 | API de alto nivel para construccion del MLP |

---

## Pipeline del Proyecto

### Fase 1 — Analisis Exploratorio y Limpieza

Se cargaron 1,000,000 filas del dataset con `nrows` y se realizo un analisis de distribuciones, valores nulos y outliers antes de cualquier transformacion.

**Valores nulos detectados:**
- `dropoff_longitude` y `dropoff_latitude`: 10 registros (0.001%) — eliminados directamente.

**Outliers tratados:**

| Variable | Criterio de eliminacion | Registros afectados |
|---|---|---|
| `fare_amount` | Tarifa <= $0 o > $200 | 105 |
| Coordenadas geograficas | Fuera del bounding box de NYC | ~20,000 |
| `passenger_count` | Igual a 0 o mayor a 7 | 3,566 |
| `distance_km` | Igual a 0 o mayor a 100 km | ~10,000 |

**Resultado de la limpieza:**

| Etapa | Filas restantes |
|---|---|
| Dataset inicial | 1,000,000 |
| Tras eliminar nulos | 999,990 |
| Tras limpiar fare_amount | 999,885 |
| Tras limpieza geografica | 978,475 |
| Tras limpiar pasajeros | 974,993 |
| Tras filtrar distancias | **964,769** |

Las distribuciones antes y despues de la limpieza demuestran el impacto de los filtros aplicados:

![Distribuciones antes de limpieza](notebook/Limpieza%20y%20carga/distribucion_antes_limpieza.png)

![Distribuciones despues de limpieza](notebook/Limpieza%20y%20carga/distribucion_despues_limpieza.png)

La matriz de correlacion entre las variables originales revela que `pickup_longitude` presenta la mayor correlacion con `fare_amount` (0.42), mientras que `passenger_count` es practicamente independiente de la tarifa (0.01).

![Matriz de correlacion](notebook/Limpieza%20y%20carga/correlacion.png)

---

### Fase 2 — Ingenieria de Caracteristicas

Se crearon 12 variables nuevas a partir de los datos originales para mejorar la capacidad predictiva de los modelos.

**Distancia Haversine:** calculo de la distancia geodesica entre origen y destino considerando la curvatura terrestre (R = 6,371 km). Es la variable mas importante del proyecto, con un 83% de importancia en el Random Forest.

**Variables temporales:** extraccion de `hour`, `day_of_week`, `month` y `year` del campo `pickup_datetime`, permitiendo capturar patrones de demanda por hora pico, dias de la semana y estacionalidad anual.

**Flags de aeropuertos:** variables binarias (0/1) que indican si el origen o destino del viaje se encuentra dentro del radio de JFK, LaGuardia o Newark. Relevante porque los viajes a aeropuertos tienen tarifas fijas o significativamente mas altas.

**Dataset final:** 964,769 registros x 19 columnas (7 originales + 12 nuevas).

![Distribucion de variables nuevas](notebook/Limpieza%20y%20carga/feature_engineering.png)

---

### Fase 3 — Particion Train/Test

Division del dataset en 80% entrenamiento y 20% prueba usando `train_test_split` con `random_state=42` para garantizar reproducibilidad.

| Conjunto | Filas | Proporcion |
|---|---|---|
| Train | 771,815 | 80% |
| Test | 192,954 | 20% |

Se utilizaron las 16 variables resultantes de la ingenieria de caracteristicas como features de entrada, y `fare_amount` como variable objetivo.

---

### Fase 4 — Experimentacion con Modelos

Se entrenaron tres modelos desde cero, sin uso de modelos pre-entrenados.

---

#### Modelo 1: Regresion Lineal Multivariada

Modelo base del proyecto. Ajusta una combinacion lineal de las 16 variables de entrada mediante minimos cuadrados ordinarios. No requiere seleccion de hiperparametros.

**Metricas:**

| Conjunto | MAE | RMSE | R2 |
|---|---|---|---|
| Train | 2.1444 | 4.2039 | 0.8049 |
| Test | 2.1469 | 4.2267 | 0.8017 |

La diferencia minima entre train y test confirma ausencia de overfitting. Sin embargo, el modelo no captura relaciones no lineales, lo que se evidencia en los errores en tarifas de aeropuerto y viajes de tarifa alta.

![Predicciones vs Valores Reales - Regresion Lineal](notebook/Regresion%20Lineal%20Multivariada/lr_predicciones_vs_reales.png)

![Residuos - Regresion Lineal](notebook/Regresion%20Lineal%20Multivariada/lr_residuos.png)

---

#### Modelo 2: Random Forest

Ensamble de arboles de decision con busqueda de hiperparametros mediante `GridSearchCV` con validacion cruzada de 3 folds sobre una submuestra de 100,000 registros.

**Grilla de busqueda:**

| Hiperparametro | Valores explorados |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 15, 20, None |
| `min_samples_split` | 5, 10 |

**Mejor configuracion:** `n_estimators=200`, `max_depth=None`, `min_samples_split=10`

**Metricas:**

| Conjunto | MAE | RMSE | R2 |
|---|---|---|---|
| Train | 0.9196 | 1.9957 | 0.9560 |
| Test | 1.6436 | 3.4233 | 0.8699 |

Se detecta overfitting moderado (diferencia de R2 = 0.086 entre train y test), atribuible al uso de `max_depth=None` que permite arboles sin limite de profundidad. El tiempo de entrenamiento fue de 350.8 segundos.

![Predicciones vs Valores Reales - Random Forest](notebook/Random%20Forest/rf_predicciones_vs_reales.png)

![Residuos - Random Forest](notebook/Random%20Forest/rf_residuos.png)

![Importancia de Features - Random Forest](notebook/Random%20Forest/rf_importancia_features.png)

La variable `distance_km` concentra el 83% de la importancia, confirmando que la distancia es el principal determinante de la tarifa.

---

#### Modelo 3: Red Neuronal Densa (MLP)

Arquitectura de red neuronal profunda construida con TensorFlow/Keras. Se realizo una busqueda manual de hiperparametros evaluando 6 configuraciones distintas sobre 150,000 muestras con Early Stopping.

**Configuraciones evaluadas:**

| Configuracion | Capas | Dropout | Val MAE |
|---|---|---|---|
| Shallow-NoDrop | [128, 64] | 0.0 | 1.8509 |
| **Medium-Drop10** | **[256, 128, 64]** | **0.1** | **1.6905** |
| Medium-Drop20 | [256, 128, 64] | 0.2 | 9.7623 |
| Deep-Drop20 | [512, 256, 128, 64] | 0.2 | 9.3790 |
| Deep4-Drop15 | [256, 128, 64, 32] | 0.15 | 12.1898 |
| Wide3-Drop20 | [512, 256, 128] | 0.2 | 8.8478 |

**Arquitectura final (Medium-Drop10):**

| Capa | Salida | Parametros |
|---|---|---|
| Dense(256) + ReLU + BN | (None, 256) | 4,608 |
| Dense(128) + ReLU + BN | (None, 128) | 33,280 |
| Dense(64) + ReLU + BN | (None, 64) | 8,320 |
| Dense(1) | (None, 1) | 65 |
| **Total** | | **46,465** |

Optimizador: Adam (lr=0.001). Callbacks: EarlyStopping (patience=8) y ReduceLROnPlateau (factor=0.5, patience=4). Los datos de entrada fueron normalizados con `StandardScaler` antes del entrenamiento.

**Metricas:**

| Conjunto | MAE | RMSE | R2 |
|---|---|---|---|
| Train | 1.5824 | 3.4067 | 0.8719 |
| Test | 1.5949 | 3.4674 | 0.8665 |

Diferencia de R2 entre train y test de 0.004, confirmando ausencia de overfitting. Tiempo de entrenamiento: 308 segundos en CPU.

![Curvas de Aprendizaje - MLP](notebook/Redes%20Neuronales%20Densas/mlp_curvas_aprendizaje.png)

![Predicciones vs Valores Reales - MLP](notebook/Redes%20Neuronales%20Densas/mlp_predicciones_vs_reales.png)

![Residuos - MLP](notebook/Redes%20Neuronales%20Densas/mlp_residuos.png)

---

## Comparativa Final de Modelos

| Modelo | MAE Train | MAE Test | RMSE Test | R2 Test | Overfitting | Tiempo (s) |
|---|---|---|---|---|---|---|
| Regresion Lineal | 2.1444 | 2.1469 | 4.2267 | 0.8017 | Ninguno | 0.11 |
| Random Forest | 0.9196 | 1.6436 | 3.4233 | 0.8699 | Moderado | 350.8 |
| **MLP** | **1.5824** | **1.5949** | **3.4674** | **0.8665** | **Ninguno** | **308.1** |

**Modelo seleccionado: MLP.** Ofrece el mejor balance entre precision y generalizacion. Obtiene un MAE de $1.59 en test con una diferencia minima respecto al train, confirmando ausencia de overfitting. Aunque el Random Forest logra un MAE ligeramente inferior ($1.64), presenta overfitting moderado que compromete su confiabilidad con datos nuevos. La Regresion Lineal queda como modelo base de referencia.

---

## Prueba de Usuario

Se implemento la funcion `predecir_tarifa()` en la celda final del notebook. Recibe coordenadas de origen y destino, fecha/hora y numero de pasajeros, calcula internamente la distancia Haversine y los flags de aeropuerto, y devuelve la prediccion de los tres modelos.

**Ejemplo de uso:**

```python
predecir_tarifa(
    pickup_lat=40.7580, pickup_lon=-73.9855,
    dropoff_lat=40.6413, dropoff_lon=-73.7781,
    datetime_str='2016-03-20 08:00:00',
    passengers=2
)
```

**Salida:**
```
Origen      : (40.758, -73.9855)
Destino     : (40.6413, -73.7781)
Distancia   : 21.77 km
Fecha/Hora  : 2016-03-20 08:00:00
Pasajeros   : 2
----------------------------------------
Regresion Lineal : $55.98
Random Forest    : $56.13
MLP              : $57.24
```

---

## Ejecucion del Notebook

```bash
# Activar el entorno virtual
source .venv312/bin/activate

# Abrir Jupyter
jupyter notebook notebook/proyecto1.ipynb
```

En VS Code: seleccionar el kernel **".venv312 (Python 3.12.5)"** y ejecutar todas las celdas en orden (`Run All`).

**Nota:** el dataset `train.csv` debe estar ubicado en `data/train.csv`. No se incluye en el repositorio por su tamano (5 GB+). Descargarlo desde [Kaggle](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data).
