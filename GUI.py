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

        self.setWindowTitle("Digit Recognition System")
        self.setFixedSize(700, 750)

        #title
        self.title = QLabel("Handwritten Digit Recognition", self)
        self.title.setFont(QFont("Arial", 18, QFont.Bold))
        self.title.setGeometry(0, 20, 700, 40)
        self.title.setAlignment(Qt.AlignCenter)

        #path
        self.path = QLineEdit(self)
        self.path.setPlaceholderText("Select or enter image path...")
        self.path.setGeometry(100, 90, 500, 40)

        #buttons
        self.folderbutton = QPushButton('Browse', self)
        self.folderbutton.setGeometry(180, 150, 150, 40)
        self.folderbutton.clicked.connect(self.open_file)
        self.submitbutton = QPushButton("Predict", self)
        self.submitbutton.setGeometry(370, 150, 150, 40)
        self.submitbutton.clicked.connect(self.predict)

        #section headings
        self.image_label = QLabel("Processed Image", self)
        self.image_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.image_label.setGeometry(80, 210, 200, 30)

        self.conf_label = QLabel("Confidence Scores", self)
        self.conf_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.conf_label.setGeometry(380, 210, 200, 30)

        #img
        self.display_label = QLabel(self)
        self.display_label.setGeometry(50, 250, 280, 280)
        self.display_label.setStyleSheet(
            "background-color: black; border: 2px solid #555;"
        )
        self.display_label.setAlignment(Qt.AlignCenter)

        #Confidence score
        self.confidence_box = QTextEdit(self)
        self.confidence_box.setGeometry(350, 250, 300, 280)
        self.confidence_box.setReadOnly(True)
        self.confidence_box.setStyleSheet(
            "font-family: monospace; font-size: 13px;"
        )

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
            results, combined_view = predict_digit(
                path_text, self.pca, self.rf
            )

            # Display image
            h, w = combined_view.shape
            qimg = QImage(combined_view.data, w, h, w, QImage.Format_Grayscale8)
            self.display_label.setPixmap(
                QPixmap.fromImage(qimg).scaled(260, 260, Qt.KeepAspectRatio)
            )

            full_prediction = "".join([str(r['digit']) for r in results])

            text = ""
            for idx, r in enumerate(results):
                text += f"Digit {idx + 1}:\n"
                for i in range(10):
                    bar = "█" * int(r['probs'][i] * 25)
                    text += f"{i}: {r['probs'][i]*100:5.1f}% {bar}\n"
                text += "\n"

            self.confidence_box.setText(text)

            avg_conf = sum(r['conf'] for r in results) / len(results)

            QMessageBox.information(
                self,
                "Prediction",
                f"Predicted: {full_prediction}\nConfidence: {avg_conf:.1f}%"
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
