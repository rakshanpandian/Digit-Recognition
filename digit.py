import cv2
import numpy as np


def shift_to_center_mass(img):
    moments = cv2.moments(img)

    if moments["m00"] == 0:
        return img

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    shift_x = 14 - cx
    shift_y = 14 - cy

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (28, 28))


def preprocess_and_segment(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)    #Making sure it's a white digit on black

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No digits found")

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    digits = []

    for c in contours:
        if cv2.contourArea(c) < 50:
            continue

        x, y, w, h = cv2.boundingRect(c)

        pad = 4
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(binary.shape[1] - x, w + 2 * pad)
        h = min(binary.shape[0] - y, h + 2 * pad)

        digit = binary[y:y+h, x:x+w]

        digit = cv2.GaussianBlur(digit, (3, 3), 0)  #Slight gaussian blur fr smoothening the image

        #to 20x20
        target = 20
        if h > w:
            new_h = target
            new_w = int(w * target / h)
        else:
            new_w = target
            new_h = int(h * target / w)

        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        #Pad to 28x28
        padded = np.zeros((28, 28), dtype=np.uint8)

        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

        #Center mass alignment
        padded = shift_to_center_mass(padded)

        digits.append(padded)

    return digits


def predict_digit(image_path, pca, rf):
    digit_images = preprocess_and_segment(image_path)

    results = []
    processed_images = []

    for img in digit_images:
        vec = (img / 255.0).reshape(1, 784).astype(np.float32)

        transformed = pca.transform(vec)
        probs = rf.predict_proba(transformed)[0]

        results.append({
            "digit": int(np.argmax(probs)),
            "conf": float(np.max(probs) * 100),
            "probs": probs
        })

        processed_images.append(img)

    combined_view = np.hstack(processed_images)

    return results, combined_view
