#!/usr/bin/env python
import os
import time
import argparse
from typing import Optional

import cv2
import numpy as np
import onnxruntime  # type: ignore


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    image: np.ndarray,
    score_th: Optional[float] = None,
) -> np.ndarray:
    # ONNX Input Size
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process: Resize, BGR->RGB, Normalize, Transpose, float32 cast
    input_image = cv2.resize(image, dsize=(input_width, input_height))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_image = (input_image / 255.0 - mean) / std  # type: ignore
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})

    # Post process: Squeeze, Sigmoid, Multiply by 255, uint8 cast
    mask = np.squeeze(result[-1])
    mask = sigmoid(mask)
    if score_th is not None:
        mask = np.where(mask < score_th, 0, 1)
    mask *= 255
    mask = mask.astype('uint8')

    return mask


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--movie', type=str, default=None)
    parser.add_argument(
        '--model',
        type=str,
        default='model/birefnet_1024x1024.onnx',
    )
    parser.add_argument('--score_th', type=float, default=None)

    args = parser.parse_args()
    model_path = args.model
    score_th = args.score_th

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv2.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        mask = run_inference(
            onnx_session,
            frame,
            score_th,
        )

        elapsed_time = time.time() - start_time

        # Resize
        mask = cv2.resize(
            mask,
            dsize=(frame.shape[1], frame.shape[0]),
        )

        # Mask extract
        temp_image = np.zeros(frame.shape, dtype=np.uint8)
        temp_image[:] = (255, 255, 255)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        debug_image = np.where(mask, frame, temp_image)

        # Inference elapsed time
        elapsed_time_text = 'Elapsed time: '
        elapsed_time_text += str(round((elapsed_time * 1000), 1))
        elapsed_time_text += 'ms'
        cv2.putText(debug_image, elapsed_time_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('BiRefNet : Input', frame)
        cv2.imshow('BiRefNet : Output', mask)
        cv2.imshow('BiRefNet : Debug', debug_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def download_file(
    url: str,
    save_path: str,
    retries: int = 10,
) -> None:
    import requests  # type: ignore
    from tqdm import tqdm  # type: ignore

    print('Download:', save_path)

    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with open(save_path, 'wb') as file, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=save_path,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))
            return
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                raise


if __name__ == '__main__':
    if not os.path.exists('./model/birefnet_1024x1024.onnx'):
        url = 'https://github.com/Kazuhito00/BiRefNet-ONNX-Sample/releases/download/v0.0.1/birefnet_1024x1024.onnx'
        save_path = './model/birefnet_1024x1024.onnx'
        download_file(url, save_path)

    main()
