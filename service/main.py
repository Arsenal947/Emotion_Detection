from fastapi import FastAPI
import onnxruntime as rt
from service.api.api import main_router

app = FastAPI(project_name= "Emotion Detection")
app.include_router(main_router)

providers = ['CPUExecutionProvider']
m_q = rt.InferenceSession('vit_onnx.onnx', providers = providers)

@app.get('/')
async def root():
    return {'hello':'hii'}
