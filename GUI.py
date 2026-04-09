import sys
import os
import joblib
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit,
                             QPushButton, QMessageBox, QFileDialog, QTextEdit)
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt
from model_train import load_or_train_model

from digit import predict_digit


class MainWindow(QMainWindow):
    def __init__(self, scaler, pca, rf):
        super().__init__()
        self.scaler = scaler
        self.pca = pca
        self.rf = rf

        self.setWindowTitle("Digit Recognition")
        self.setFixedSize(600, 780)

        self.labelpath = QLabel("Path", self)
        self.labelpath.setFont(QFont("Times New Roman", 20))
        self.labelpath.setGeometry(0, 120, 600, 50)
        self.labelpath.setAlignment(Qt.AlignCenter)

        self.path = QLineEdit(self)
        self.path.setPlaceholderText("Enter Path")
        self.path.setGeometry(100, 180, 400, 50)

        self.folderbutton = QPushButton('Open Image', self)
        self.folderbutton.setGeometry(150, 250, 140, 50)
        self.folderbutton.clicked.connect(self.open_file)

        self.submitbutton = QPushButton("Predict", self)
        self.submitbutton.setGeometry(310, 250, 140, 50)
        self.submitbutton.clicked.connect(self.predict)

        self.display_label = QLabel(self)
        self.display_label.setGeometry(60, 330, 480, 250)
        self.display_label.setStyleSheet("background-color: black; border: 1px solid white;")
        self.display_label.setAlignment(Qt.AlignCenter)

        self.confidence_box = QTextEdit(self)
        self.confidence_box.setGeometry(100, 600, 400, 150)
        self.confidence_box.setReadOnly(True)
        self.confidence_box.setStyleSheet("font-family: monospace; font-size: 12px;")

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.path.setText(file_path)

    def predict(self):
        path_text = self.path.text()
        if not os.path.exists(path_text):
            QMessageBox.critical(self, "Error", "Invalid Path")
            return

        try:
            # predict_digit now handles segmentation for multiple digits
            results, combined_view = predict_digit(
                path_text, self.scaler, self.pca, self.rf
            )

            h, w = combined_view.shape
            qimg = QImage(combined_view.data, w, h, w, QImage.Format_Grayscale8)
            self.display_label.setPixmap(
                QPixmap.fromImage(qimg).scaled(460, 200, Qt.KeepAspectRatio)
            )

            full_prediction = "".join([str(r['digit']) for r in results])

            report = f"Sequence: {full_prediction}\n" + "-" * 30 + "\n"
            for i, r in enumerate(results):
                report += f"Digit {i + 1} ({r['digit']}): {r['conf']:.1f}%\n"
                for j in range(10):
                    bar = "█" * int(r['probs'][j] * 20)
                    report += f"  {j}: {bar} {r['probs'][j] * 100:4.1f}%\n"
                report += "\n"

            self.confidence_box.setText(report)
            QMessageBox.information(self, "Prediction", f"Full Number Predicted: {full_prediction}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def main():
    model_path = "digit_rf_stable_v1.pkl"

    if os.path.exists(model_path):
        # Load from file to bypass TensorFlow DLL initialization error
        scaler, pca, rf = joblib.load(model_path)
    else:
        # Only runs module generation if the pkl doesn't exist
        scaler, pca, rf = load_or_train_model()

    app = QApplication(sys.argv)
    window = MainWindow(scaler, pca, rf)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
