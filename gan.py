import onnxruntime
from PIL import Image
import numpy as np

def prepare_image(image, SIZE=512):
    """
    Подготавливает изображение для входа в модель.
    - Изменяет размер
    - Нормализует пиксели
    - Преобразует в тензор формата (1, C, H, W)
    """
    assert isinstance(image, Image.Image), "Input image must be a PIL Image"
    
    image = np.array(image.resize((SIZE, SIZE)))
    assert image.shape == (SIZE, SIZE, 3), "Resized image must have shape (SIZE, SIZE, 3)"
    
    image = image / 255.0
    image = (image - 0.5) / 0.5
    image = np.transpose(image, [2, 0, 1])
    
    return np.expand_dims(image.astype(np.float32), 0)

def postprocess_image(image):
    """
    Постобработка выходного изображения.
    - Преобразует обратно в (H, W, C)
    - Де-нормализует пиксели
    """
    assert isinstance(image, np.ndarray), "Input must be a NumPy array"
    
    image = np.transpose(image[0], [1, 2, 0])
    image = (image * 0.5 + 0.5) * 255
    
    return image

def run_gan(image):
    """
    Запускает ONNX-модель GAN на входном изображении и возвращает обработанный результат.
    """
    assert isinstance(image, Image.Image), "Input image must be a PIL Image"
    
    shapes = image.size
    image = prepare_image(image)
    ort_session = onnxruntime.InferenceSession("./my_gan_A.onnx")
    
    # Выполняем предсказание
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    
    output_image = postprocess_image(ort_outs[0]).astype(np.uint8)
    assert output_image.shape[2] == 3, "Output image must have 3 channels"
    
    return Image.fromarray(output_image).resize(shapes[:2])