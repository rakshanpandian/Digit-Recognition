import cv2
import numpy as np


def shift_to_center_mass(img):
    moments = cv2.moments(img)
    if moments["m00"] == 0:
        return img
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    shift_x, shift_y = 14 - cx, 14 - cy
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (28, 28))


def predict_digit(image_path, scaler, pca, rf):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")

    h_orig, w_orig = image.shape[:2]
    image = image[int(h_orig * 0.08):int(h_orig * 0.92), int(w_orig * 0.08):int(w_orig * 0.92)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Thicken the lines slightly to match MNIST 'bold' style
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_contours = [c for c in contours if cv2.contourArea(c) > 100]
    digit_contours = sorted(digit_contours, key=lambda c: cv2.boundingRect(c)[0])

    if not digit_contours:
        raise ValueError("No digits detected")

    results = []
    processed_images = []

    for c in digit_contours:
        x, y, w, h = cv2.boundingRect(c)
        digit_roi = binary[y:y + h, x:x + w]

        target = 20
        ratio = target / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        digit_res = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        padded = np.zeros((28, 28), dtype=np.uint8)
        padded[(28 - new_h) // 2:(28 - new_h) // 2 + new_h, (28 - new_w) // 2:(28 - new_w) // 2 + new_w] = digit_res

        padded = shift_to_center_mass(padded)
        processed_images.append(padded)

        # Normalize and Scale
        vec = (padded / 255.0).reshape(1, 784).astype(np.float32)

        # Apply Scaler before PCA
        scaled_vec = scaler.transform(vec)
        transformed = pca.transform(scaled_vec)

        probs = rf.predict_proba(transformed)[0]

        results.append({
            "digit": np.argmax(probs),
            "conf": np.max(probs) * 100,
            "probs": probs
        })

    combined_view = np.hstack(processed_images)

    return results, combined_view
