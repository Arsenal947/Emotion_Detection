import cv2
import numpy as np
import time 
import service.main as s
import onnxruntime as rt

def emotions_detecter(image_array):
    if len(image_array.shape)== 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    t1 = time.time()
    test_image = cv2.resize(image_array, (256, 256))
    im = np.float32(test_image)

    image_array = np.expand_dims(im, axis = 0)


    onnx_pred = s.m_q.run(['dense'], {"input_2": image_array})
    time_elapsed = time.time()-t1
    emotion = ''
    if np.argmax(onnx_pred[0][0]) == 0:
        emotion = 'angry'
    elif np.argmax(onnx_pred[0][0]) == 1:
        emotion = 'sad'
    else:
        emotion = 'happy'
    return {
        "emotion": emotion,
        "time_elapsed": str(time_elapsed)
    }