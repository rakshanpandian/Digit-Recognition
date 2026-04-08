import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit,
                             QPushButton, QMessageBox, QFileDialog, QTextEdit)
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt

from model_train import load_or_train_model
from digit import predict_digit


class MainWindow(QMainWindow):
    def __init__(self, pca, rf):
        super().__init__()
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
        self.display_label.setGeometry(175, 330, 250, 250)
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
            predicted, confidence, probs, padded = predict_digit(
                path_text, self.pca, self.rf
            )

            # Show image
            h, w = padded.shape
            qimg = QImage(padded.data, w, h, w, QImage.Format_Grayscale8)
            self.display_label.setPixmap(
                QPixmap.fromImage(qimg).scaled(250, 250, Qt.KeepAspectRatio)
            )

            # Confidence bars
            text = "Confidence scores:\n"
            for i in range(10):
                bar = "█" * int(probs[i] * 30)
                text += f"{i}: {probs[i]*100:5.1f}% {bar}\n"

            self.confidence_box.setText(text)

            QMessageBox.information(
                self,
                "Prediction",
                f"Predicted Digit: {predicted}\nConfidence: {confidence:.1f}%"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def main():
    pca, rf = load_or_train_model()

    app = QApplication(sys.argv)
    window = MainWindow(pca, rf)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
