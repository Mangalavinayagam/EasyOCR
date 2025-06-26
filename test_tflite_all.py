#!/usr/bin/env python3
"""
easyocr_tflite_pipeline.py

End-to-end evaluation of EasyOCR TFLite detector + recognizer without OpenCV:
- Runs detector on full image to get text/link score maps.
- Post-processes maps via pure-Python BFS for connected components (no OpenCV).
- Crops each region and runs recognizer TFLite model on each crop.
- Optionally decodes via CTC using provided labels.txt.

Usage:
    python3 easyocr_tflite_pipeline.py \
      --detector_model detector.tflite \
      --recognizer_model recognizer.tflite \
      --image input.jpg \
      [--labels labels.txt] \
      [--text_threshold 0.7] [--link_threshold 0.4] \
      [--img_height 32] [--width_divisor 1]

Notes:
- Requires: numpy, Pillow, tflite-runtime or tensorflow.
- No OpenCV needed.
"""
import argparse
import sys
import math
from collections import deque
from PIL import Image
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path='/home/plap079/Programs/Python/EasyOCR_EasyOCRDetector.tflite')

    except ImportError:
        Interpreter = None

try:
    from PIL import Image
except ImportError:
    Image = None

def load_tflite_model(model_path: str):
    if Interpreter is None:
        print("Error: No TFLite Interpreter available. Install tflite-runtime or tensorflow.", file=sys.stderr)
        sys.exit(1)
    try:
        interpreter = Interpreter(model_path=model_path)
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}", file=sys.stderr)
        sys.exit(1)
    interpreter.allocate_tensors()
    return interpreter

def print_model_details(interpreter, name):
    print(f"\n{name} model I/O details:")
    for i, inp in enumerate(interpreter.get_input_details()):
        print(f" Input {i}: {inp}")
    for i, out in enumerate(interpreter.get_output_details()):
        print(f" Output {i}: {out}")
    print()

def preprocess_detector_image(orig_img: Image.Image, input_shape, input_dtype, input_quant):
    """
    Preprocess full image for detector:
    - Resize to model input size.
    - Normalize or quantize.
    Returns (input_data numpy array, resized_size tuple (w,h)).
    """
    if Image is None:
        print("Error: PIL not installed.", file=sys.stderr)
        sys.exit(1)
    if len(input_shape) != 4:
        print(f"Error: Detector input shape not 4D: {input_shape}", file=sys.stderr)
        sys.exit(1)
    batch, d1, d2, d3 = input_shape
    # Determine layout and target dims
    # NHWC if d3 in {1,3}, else NCHW if d1 in {1,3}
    if d3 in (1,3):
        layout = 'NHWC'
        target_h, target_w = d1, d2
        channels = d3
    elif d1 in (1,3):
        layout = 'NCHW'
        target_h, target_w = d2, d3
        channels = d1
    else:
        print(f"Cannot infer detector layout from shape {input_shape}", file=sys.stderr)
        sys.exit(1)
    # Resize
    resized = orig_img.resize((target_w, target_h))
    arr = np.array(resized.convert("RGB"), dtype=np.float32)  # (H,W,3)
    if channels == 1:
        arr = np.array(resized.convert("L"), dtype=np.float32)[..., None]
    # Arrange layout
    if layout == 'NHWC':
        data = arr  # (H,W,C)
    else:  # NCHW
        data = arr.transpose((2,0,1))  # (C,H,W)
    # Normalize or quantize
    scale, zero_point = input_quant
    if np.issubdtype(input_dtype, np.floating):
        data = data / 255.0
        data = data.astype(input_dtype)
    else:
        data = data / 255.0
        if scale != 0:
            data = data / scale + zero_point
            data = np.clip(data, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
        data = data.astype(input_dtype)
    # Add batch
    data = np.expand_dims(data, axis=0)
    if tuple(data.shape) != tuple(input_shape):
        print(f"Error: Detector preprocessed shape {data.shape} != expected {input_shape}", file=sys.stderr)
        sys.exit(1)
    return data, (target_w, target_h)

def run_detector(interpreter, input_data):
    inp = interpreter.get_input_details()[0]
    interpreter.set_tensor(inp['index'], input_data)
    interpreter.invoke()
    out0 = interpreter.get_output_details()[0]
    return interpreter.get_tensor(out0['index'])  # shape (1,h2,w2,2)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def postprocess_score_link(raw_maps, text_threshold, link_threshold):
    """
    raw_maps: numpy array shape (1,h,w,2)
    Returns binary combined mask shape (h,w) dtype uint8.
    """
    maps = raw_maps[0]
    score_map = maps[:,:,0]
    link_map  = maps[:,:,1]
    # Apply sigmoid if outside [0,1]
    if score_map.min() < 0 or score_map.max() > 1:
        score_map = sigmoid(score_map)
    if link_map.min() < 0 or link_map.max() > 1:
        link_map = sigmoid(link_map)
    text_mask = (score_map > text_threshold).astype(np.uint8)
    link_mask = (link_map  > link_threshold).astype(np.uint8)
    combined = text_mask | link_mask
    return combined  # 0/1

def find_connected_boxes(mask, min_area=10):
    """
    mask: numpy array shape (h,w) values 0/1.
    Returns list of boxes in mask coords: (x_min,y_min,x_max,y_max).
    Uses BFS for connected components (8-connectivity).
    """
    h, w = mask.shape
    visited = np.zeros((h,w), dtype=bool)
    boxes = []
    # neighbor offsets for 8-connectivity
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y in range(h):
        for x in range(w):
            if mask[y,x] and not visited[y,x]:
                # start BFS
                q = deque()
                q.append((y,x))
                visited[y,x] = True
                xs = [x]; ys = [y]
                while q:
                    cy, cx = q.popleft()
                    for dy,dx in neigh:
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny,nx] and not visited[ny,nx]:
                                visited[ny,nx] = True
                                q.append((ny,nx))
                                xs.append(nx); ys.append(ny)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                area = (x_max - x_min + 1)*(y_max - y_min + 1)
                if area >= min_area:
                    boxes.append((x_min, y_min, x_max, y_max))
    return boxes

def map_box_mask_to_original(box, mask_size, resized_size, orig_size):
    """
    box: (x_min,y_min,x_max,y_max) in mask coords (mask is half resolution of resized image).
    mask_size: (mask_w, mask_h) == raw_maps.shape[2], shape[1].
    resized_size: (resized_w, resized_h)
    orig_size: (orig_w, orig_h)
    Returns box in original image coords: (x1,y1,x2,y2).
    """
    x_min, y_min, x_max, y_max = box
    mask_h, mask_w = mask_size[1], mask_size[0]  # careful: mask shape is (h,w)
    resized_w, resized_h = resized_size
    orig_w, orig_h = orig_size
    # The model output spatial dims are half of resized dims (for EasyOCR detector).
    # So map mask coords to resized coords by factor 2:
    # x_resized_min = x_min*2, x_resized_max = (x_max+1)*2 - 1? We'll include full region: use (x_max+1)*2
    x1_r = x_min * 2
    x2_r = (x_max + 1) * 2
    y1_r = y_min * 2
    y2_r = (y_max + 1) * 2
    # Now map resized coords to original:
    fx = orig_w / resized_w
    fy = orig_h / resized_h
    x1 = int(round(x1_r * fx))
    x2 = int(round(x2_r * fx))
    y1 = int(round(y1_r * fy))
    y2 = int(round(y2_r * fy))
    # Clip
    x1 = max(0, min(x1, orig_w-1))
    x2 = max(1, min(x2, orig_w))
    y1 = max(0, min(y1, orig_h-1))
    y2 = max(1, min(y2, orig_h))
    if x2 <= x1: x2 = min(orig_w, x1+1)
    if y2 <= y1: y2 = min(orig_h, y1+1)
    return (x1, y1, x2, y2)

def preprocess_recognizer_image(crop_img: Image.Image, input_shape, input_dtype, input_quant,
                                override_height=None, override_width=None, width_divisor=1):
    """
    Preprocess a cropped PIL Image for recognizer model.
    Returns numpy array shape input_shape.
    """
    if Image is None:
        print("Error: PIL not installed.", file=sys.stderr)
        sys.exit(1)
    if len(input_shape) != 4:
        print(f"Error: Recognizer input shape not 4D: {input_shape}", file=sys.stderr)
        sys.exit(1)
    batch, d1, d2, d3 = input_shape
    # Detect layout
    if d3 in (1,3):
        layout = 'NHWC'
        channels = d3
        target_h, target_w = d1, d2
    elif d1 in (1,3):
        layout = 'NCHW'
        channels = d1
        target_h, target_w = d2, d3
    else:
        layout = 'NHWC'
        channels = 3
        target_h, target_w = d1, d2
    # Handle dynamic height
    if target_h is None or target_h <= 0:
        if override_height is None:
            print("Error: Recognizer dynamic height; use --img_height.", file=sys.stderr)
            sys.exit(1)
        target_h = override_height
    # Handle dynamic width
    if target_w is None or target_w <= 0:
        if override_width is not None:
            target_w = override_width
        else:
            orig_w, orig_h = crop_img.size
            new_w = math.ceil(orig_w * (target_h / orig_h))
            if width_divisor and width_divisor > 1:
                new_w = math.ceil(new_w / width_divisor) * width_divisor
            target_w = new_w
    # Resize
    resized = crop_img.resize((target_w, target_h))
    arr = np.array(resized.convert("RGB"), dtype=np.float32)
    if channels == 1:
        arr = np.array(resized.convert("L"), dtype=np.float32)[..., None]
    if layout == 'NHWC':
        data = arr
    else:
        data = arr.transpose((2,0,1))
    # Normalize or quantize
    scale, zero_point = input_quant
    if np.issubdtype(input_dtype, np.floating):
        data = data / 255.0
        data = data.astype(input_dtype)
    else:
        data = data / 255.0
        if scale != 0:
            data = data / scale + zero_point
            data = np.clip(data, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
        data = data.astype(input_dtype)
    data = np.expand_dims(data, axis=0)
    if tuple(data.shape) != tuple(input_shape):
        print(f"Error: Recognizer preprocessed shape {data.shape} != expected {input_shape}", file=sys.stderr)
        sys.exit(1)
    return data

def load_labels(labels_path: str):
    labels = []
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                labels.append(line.strip('\n').rstrip('\r'))
    except Exception as e:
        print(f"Error opening labels file '{labels_path}': {e}", file=sys.stderr)
        sys.exit(1)
    return labels

def ctc_greedy_decode(logits: np.ndarray, labels: list, blank_index: int):
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    indices = np.argmax(logits, axis=-1)
    decoded = []
    prev = None
    for idx in indices:
        if idx == prev or idx == blank_index:
            prev = idx
            continue
        if 0 <= idx < len(labels):
            decoded.append(labels[idx])
        else:
            decoded.append('?')
        prev = idx
    return ''.join(decoded)

def run_recognizer(interpreter, crop_img, rec_input_shape, rec_input_dtype, rec_input_quant,
                   override_height=None, override_width=None, width_divisor=1,
                   labels=None, blank_index=None):
    data = preprocess_recognizer_image(crop_img, rec_input_shape, rec_input_dtype, rec_input_quant,
                                       override_height=override_height, override_width=override_width,
                                       width_divisor=width_divisor)
    inp = interpreter.get_input_details()[0]
    interpreter.set_tensor(inp['index'], data)
    interpreter.invoke()
    outdet = interpreter.get_output_details()[0]
    out = interpreter.get_tensor(outdet['index'])
    # Print raw stats
    try:
        print(f"    Recognizer raw output shape: {out.shape}, min={out.min():.6f}, max={out.max():.6f}, mean={out.mean():.6f}")
    except:
        print(f"    Recognizer raw output shape: {out.shape}")
    text = None
    if labels is not None:
        if blank_index is None:
            blank_index = len(labels)-1
        text = ctc_greedy_decode(out, labels, blank_index)
        print(f"    Decoded text: '{text}'")
    return text

def main():
    parser = argparse.ArgumentParser(description="EasyOCR TFLite pipeline (detector -> recognizer) without OpenCV.")
    parser.add_argument("--detector_model", required=True, help="Path to detector .tflite model")
    parser.add_argument("--recognizer_model", required=True, help="Path to recognizer .tflite model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--labels", help="Path to labels.txt for recognizer decoding (optional)")
    parser.add_argument("--text_threshold", type=float, default=0.7, help="Detector text threshold")
    parser.add_argument("--link_threshold", type=float, default=0.4, help="Detector link threshold")
    parser.add_argument("--img_height", type=int, help="Recognizer input height if dynamic")
    parser.add_argument("--width_divisor", type=int, default=1, help="Round recognizer width to multiple")
    args = parser.parse_args()

    if Image is None:
        print("Error: PIL not installed.", file=sys.stderr)
        sys.exit(1)

    # Load original image
    try:
        orig_img = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error opening image '{args.image}': {e}", file=sys.stderr)
        sys.exit(1)
    orig_w, orig_h = orig_img.size

    # Load detector
    det_interp = load_tflite_model(args.detector_model)
    print_model_details(det_interp, "Detector")
    det_inp = det_interp.get_input_details()[0]
    det_shape = det_inp["shape"].tolist()
    det_dtype = np.dtype(det_inp["dtype"])
    det_quant = det_inp.get("quantization", (0.0,0))

    det_input_data, (resized_w, resized_h) = preprocess_detector_image(orig_img, det_shape, det_dtype, det_quant)
    raw_maps = run_detector(det_interp, det_input_data)  # shape (1,h2,w2,2)
    print("Raw Map:",raw_maps)
    # Post-process
    mask = postprocess_score_link(raw_maps, args.text_threshold, args.link_threshold)
    h2, w2 = mask.shape
    print(f"Detector mask shape: {mask.shape}, resized image: {(resized_w,resized_h)}, original: {(orig_w,orig_h)}")
    boxes_mask = find_connected_boxes(mask)
    print(f"Found {len(boxes_mask)} regions in mask coords.")
    # Load recognizer
    rec_interp = load_tflite_model(args.recognizer_model)
    print_model_details(rec_interp, "Recognizer")
    rec_inp = rec_interp.get_input_details()[0]
    rec_shape = rec_inp["shape"].tolist()
    rec_dtype = np.dtype(rec_inp["dtype"])
    rec_quant = rec_inp.get("quantization", (0.0,0))
    # Load labels if any
    labels = None
    blank_index = None
    if args.labels:
        labels = load_labels(args.labels)
        blank_index = args.blank_index if hasattr(args, 'blank_index') and args.blank_index is not None else len(labels)-1

    # For each box, map to original coords, crop, run recognizer
    for idx, box in enumerate(boxes_mask):
        box_orig = map_box_mask_to_original(box, (w2,h2), (resized_w,resized_h), (orig_w,orig_h))
        x1,y1,x2,y2 = box_orig
        # Crop
        crop = orig_img.crop((x1,y1,x2,y2))
        print(f"Region {idx}: box in original image {box_orig}, size {(x2-x1, y2-y1)}")
        if (x2-x1)<5 or (y2-y1)<5:
            print("  Skipping too-small region.")
            continue
        # Run recognizer
        run_recognizer(rec_interp, crop, rec_shape, rec_dtype, rec_quant,
                       override_height=args.img_height, override_width=None,
                       width_divisor=args.width_divisor,
                       labels=labels, blank_index=blank_index)

if __name__ == "__main__":
    main()
