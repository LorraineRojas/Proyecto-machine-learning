# Writer Identification App

Identifica el escritor de una imagen de escritura donde los usuarios dibujaron un círculo, usando EfficientNet-B0.

## Estructura del repositorio

```
Proyecto machine learning/
│
├── app.py                  
├── requirements.txt        
├── README.md
├── notebook/
   └── entrenamientoProyectoML.ipynb
└── models/
    └── effnet_b0_fold0.pth
  
```
---

## Guía de entrenamiento

### Requisitos previos

- Cuenta de **Google Colab** con GPU (T4 o superior)
- Acceso a la competencia [kaggle](https://www.kaggle.com/competitions/icdar-2026-circleid-writer-identification)

### Paso 1 — Preparar el entorno

Abre el notebook `notebook/entrenamientoProyectoML.ipynb` en Google Colab.

Ejecuta la celda de instalación de dependencias:
```python
!pip install -q kaggle timm albumentations
```

### Paso 2 — Configurar rutas en Google Drive

Ajusta la variable `BASE` al inicio del notebook según tu estructura en Drive:
```python
BASE     = '/content/drive/MyDrive/TU_CARPETA/CircleID'
DATA     = f'{BASE}/data'
IMG_DIR  = f'{DATA}/images'
```

Crea las carpetas necesarias ejecutando la celda de `os.makedirs`.

### Paso 3 — Descargar el dataset de Kaggle

1. Ve a [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Descarga el archivo `kaggle.json`
3. Ejecuta la celda de subida de credenciales en el notebook:
   ```python
   uploaded = files.upload()  # sube kaggle.json
   ```
4. Ejecuta la celda de descarga del dataset:
   ```bash
   !kaggle competitions download -c icdar-2026-circleid-writer-identification
   ```

El dataset contiene:
- `train.csv` — imágenes etiquetadas (44 escritores)
- `additional_train.csv` — imágenes extra (conocidos + desconocidos `-1`)
- `test.csv` — imágenes de evaluación sin etiqueta

### Paso 4 — Entendimiento de los datos

Ejecuta las celdas de la sección **# Entendimiento** para visualizar:
- Distribución de escritores
- Ejemplos de imágenes por escritor
- Tamaños de imagen

### Paso 5 — Preparación y aumentación

Ejecuta las celdas de **## Preparación** y **## Aumentación**.

El preprocesamiento convierte cada imagen a:
1. Escala de grises
2. Padding cuadrado con fondo blanco
3. Resize a 224×224
4. 3 canales (replicación)
5. Normalización ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Las aumentaciones de entrenamiento incluyen:
- Transformaciones afines (traslación, escala, rotación)
- Transformación elástica
- Ruido gaussiano
- Coarse dropout

### Paso 6 — Splits con Cross-Validation

Ejecuta las celdas de **## PyTorch Datasets y Loaders**.

La estrategia de splits usa **fake unknowns**:
- Los 44 escritores se dividen en 3 folds
- En cada fold, 8 escritores se retiran como "desconocidos de validación"
- El modelo entrena con los 36 escritores restantes (`NUM_CLASSES = 36`)
- Esto simula el escenario real de la competencia

El notebook entrena sobre **fold 0** (seed 42).

### Paso 7 — Entrenamiento EfficientNet-B0

Ejecuta las celdas de **## EfficientNet-B0**.

El entrenamiento se hace en dos fases:

**Fase 1 — Warm-up de la cabeza (5 épocas)**
- Solo se entrena la capa clasificadora (`classifier`)
- `lr = 1e-4`, `weight_decay = 1e-4`
- Batch size: 64

**Fase 2 — Fine-tuning completo (20 épocas)**
- Se descongelan todos los parámetros
- Learning rate diferencial:
  - Backbone: `lr = 2e-5`
  - Clasificador: `lr = 1e-4`
- Scheduler: `CosineAnnealingLR`
- Se guarda el mejor checkpoint según `val_acc`

El modelo se guarda en:
```
BASE/checkpoints/effnet_b0_fold0.pth
```

### Paso 8 — Calibración del umbral de confianza

Al final del entrenamiento, el notebook evalúa umbrales de 0.10 a 0.95 en pasos de 0.05 sobre el conjunto de validación (escritores conocidos + fake unknowns) y selecciona el que maximiza la accuracy combinada.

El umbral óptimo encontrado se imprime como `BEST_THRESHOLD`.  
Úsalo como valor inicial del slider en la app.

### Paso 9 — Exportar el modelo

Descarga el archivo `effnet_b0_fold0.pth` desde Google Drive y colócalo en la carpeta `models/` del repositorio.

---

## Guía de ejecución de la app

### Opción A — Local

```bash
# 1. Clonar el repositorio
git clone [https://github.com/LorraineRojas/Proyecto-machine-learning.git](https://github.com/LorraineRojas/Proyecto-machine-learning)

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar el modelo
# Copia effnet_b0_fold0.pth dentro de models/

# 4. Ejecutar
streamlit run app.py
```

### Opción B — Streamlit Cloud

La aplicación se encuentra desplegada sobre streamlit cloud, en el siguiente [enlace] (https://proyecto-machine-learning.streamlit.app/)

---

## Uso de la app

1. Sube una imagen de escritura  (PNG, JPG, BMP, TIFF)
2. Ajusta el **umbral de confianza** si es necesario (default: 0.50)
3. Haz clic en **✦ Identificar escritor**
4. El resultado muestra el ID del escritor y el porcentaje de confianza

Si la confianza está por debajo del umbral, el modelo responde **Desconocido** — esto indica que el escritor probablemente no está en el conjunto de entrenamiento.

---

## Dependencias

### App (`requirements.txt`)
| Paquete | Versión mínima |
|---|---|
| streamlit | 1.35.0 |
| torch | 2.0.0 |
| timm | 1.0.0 |
| opencv-python-headless | 4.8.0 |
| numpy | 1.24.0 |
| Pillow | 9.0.0 |

### Entrenamiento (`requirements_training.txt`)
| Paquete | Uso |
|---|---|
| torch + torchvision | Framework de deep learning |
| timm | Modelos preentrenados (EfficientNet-B0) |
| albumentations | Aumentación de imágenes |
| opencv-python | Preprocesamiento |
| scikit-learn | LabelEncoder, métricas |
| pandas | Manejo de CSVs |
| matplotlib / seaborn | Visualizaciones |
| kaggle | Descarga del dataset |

---

## Notas técnicas

- El modelo fue entrenado en **fold 0** con `random.seed(42)`
- `NUM_CLASSES = 36` (44 escritores totales − 8 fake unknowns de fold 0)
- El label map interno (`Writer #0` ... `Writer #35`) corresponde al orden de `LabelEncoder().fit(sorted(known_writers_fold0))`
- Entrenamiento realizado en Google Colab con GPU NVIDIA T4
