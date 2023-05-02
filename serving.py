import numpy as np
import gradio as gr

def sepia(input_img):
    import time
    from absl import app, flags, logging
    from absl.flags import FLAGS
    import cv2
    import numpy as np
    import tensorflow as tf
    from yolov3_tf2.models import (
        YoloV3, YoloV3Tiny
    )
    from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
    from yolov3_tf2.utils import draw_outputs
    import requests
    import json

    img_raw = tf.image.decode_image(open(input_img, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, 256)
    print(img.shape)
    img = img.numpy().tolist()

    data = json.dumps({"inputs": img})
    model_serve_url = "input_your_endpoint"

    res_svc = requests.post(model_serve_url, data)
    class_names = ['negative', 'positive']

    output_data = json.loads(res_svc.text)
    boxes = output_data['outputs']['yolo_nms']
    scores = output_data['outputs']['yolo_nms_1']
    classes = output_data['outputs']['yolo_nms_2']
    nums = output_data['outputs']['yolo_nms_3']

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (np.array(boxes), scores, classes, nums), class_names)

    return img


demo = gr.Interface(sepia, gr.Image(type="jpg"), "image")
demo.launch(share=True)