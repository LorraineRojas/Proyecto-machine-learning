# Writer Identification App

Aplicación Streamlit para identificar escritores a partir de imágenes de escritura

## Estructura del repositorio

```
circleid-app/
│
├── app.py                  ← App principal (único archivo Python que necesitas)
├── requirements.txt        ← Dependencias Python
├── README.md
│
└── models/
    ├── effnet_b0_fold0.pth
    ├── resnet50_fold0.pth
    ├── convnext_tiny_fold0.pth
    └── cnn_lstm_fold0.pth
```

**No necesitas `config.json` ni `label_mapping.json`** — todo está embebido en `app.py`.

## ⚠️ Subir los modelos a GitHub

Los `.pth` superan el límite de 100 MB de GitHub. Usa **Git LFS**:

```bash
git lfs install
git lfs track "models/*.pth"
git add .gitattributes models/
git commit -m "add models"
git push
```

## Despliegue en Streamlit Cloud

1. Sube el repo a GitHub (con modelos via LFS)
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Archivo principal: `app.py`
5. **Deploy**

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```
