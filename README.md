# Writer Identification App

Aplicación Streamlit para identificar escritores a partir de imágenes de escritura

## Estructura del repositorio

```
Proyecto machine learning/
│
├── app.py                  
├── requirements.txt        
├── README.md
│
└── models/
    ├── effnet_b0_fold0.pth
    ├── resnet50_fold0.pth
    ├── convnext_tiny_fold0.pth
    └── cnn_lstm_fold0.pth
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```
