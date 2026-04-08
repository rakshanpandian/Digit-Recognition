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


def preprocess_input(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure MNIST polarity
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No digit found")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    digit = binary[y:y+h, x:x+w]

    target = 20
    if h > w:
        new_h = target
        new_w = int(w * target / h)
    else:
        new_w = target
        new_h = int(h * target / w)

    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    padded = shift_to_center_mass(padded)

    return padded


def predict_digit(image_path, pca, rf):
    padded = preprocess_input(image_path)

    digit_vector = (padded / 255.0).reshape(1, 784).astype(np.float32)

    transformed = pca.transform(digit_vector)
    probs = rf.predict_proba(transformed)[0]

    predicted = np.argmax(probs)
    confidence = probs[predicted] * 100

    return predicted, confidence, probs, padded
