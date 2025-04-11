import cv2
import sys

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import qimage2ndarray
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from PyQt5.QtGui import QPainterPath
from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import QObject, pyqtSignal, QPointF, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtWidgets import QMainWindow, QPushButton, QHBoxLayout
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


predecessors = {"A1": [],
                "B1": ["A1"],
                "B2": ["A1"],
                "C1": ["B1", "B2"],
                "C2": ["B1", "B2"],
                "C3": ["B1", "B2"],
                "C4": ["B1", "B2"],
                "C5": ["B1", "B2"],
                "C6": ["B1", "B2"]}


def array_to_qpixmap(array, scale=10, cmap='viridis'):
    # Use Matplotlib to apply a colormap
    norm = plt.Normalize(vmin=array.min(), vmax=array.max())
    mapped_array = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(array)[:, :, :3]

    # Scale up the array for visibility
    large_array = np.kron(mapped_array, np.ones((scale, scale, 1)))
    large_array = np.uint8(255 * large_array)  # Convert to uint8

    # Convert to QImage and then QPixmap
    height, width, channels = large_array.shape
    bytes_per_line = channels * width
    qimage = QImage(large_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 6, 5)
        self.fc1 = nn.Linear(96, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  # Activation from the first conv layer
        x2 = self.pool(x1)
        x3 = F.relu(self.conv2(x2))  # Activation from the second conv layer
        x = self.pool(x3)
        x = x.view(-1, 96)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, x1, x2, x3


class SignalWrapper(QObject):
    # Define a custom signal named hoverMove
    hoverMove = pyqtSignal(str, int, int)


class InteractivePixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap, id=None, parent=None, predecessors=None):
        super().__init__(pixmap, parent)
        self.id = id
        self.signal_wrapper = SignalWrapper()
        self.signal_wrapper.hoverMove.connect(self.handle_hover_move)
        self.setAcceptHoverEvents(True)
        self.original_pixmap = pixmap.copy()  # Store the original to revert highlighting
        self.current_pixmap = pixmap.copy()
        self.currently_highlighted_pos = None
        self.predecessors = predecessors

    def mapToPixmap(self, pos):
        # Map the position from the item's coordinate system to the pixmap's coordinate system
        return self.mapToParent(pos).toPoint()

    def handle_hover_move(self, item_id, x, y):
        # This method is called when the hoverMove signal is emitted
        pass

    def hoverLeaveEvent(self, event):
        self.remove_highlight()

    def highlight_area(self, positions):
        """
        Highlight only the border of the given positions (a list of coordinate tuples).
        Assumes positions define a rectangular area.
        """
        if not positions:
            return

        if isinstance(positions, tuple):
            positions = [positions]

        # Extract the boundary of the rectangle
        min_x = int(min(positions, key=lambda x: x[0])[0])
        max_x = int(max(positions, key=lambda x: x[0])[0]) + 10
        min_y = int(min(positions, key=lambda x: x[1])[1])
        max_y = int(max(positions, key=lambda x: x[1])[1]) + 10

        # Check if the currently highlighted position is different
        if self.currently_highlighted_pos != positions:
            self.currently_highlighted_pos = positions
            temp_pixmap = self.original_pixmap.copy()
            painter = QPainter(temp_pixmap)
            painter.setPen(QPen(QColor("#DC143C"), 2))  # Set pen width and color

            # Draw the perimeter of the rectangle
            # Top border
            painter.drawLine(min_x, min_y, max_x, min_y)
            # Bottom border
            painter.drawLine(min_x, max_y, max_x, max_y)
            # Left border
            painter.drawLine(min_x, min_y, min_x, max_y)
            # Right border
            painter.drawLine(max_x, min_y, max_x, max_y)

            painter.end()

            self.setPixmap(temp_pixmap)  # Update the pixmap
            self.update()

    def remove_highlight(self):
        if self.currently_highlighted_pos is not None:
            self.setPixmap(self.original_pixmap.copy())
            self.update()
            self.currently_highlighted_pos = None
        for item in self.predecessors:
            item.remove_highlight()

    def hoverMoveEvent(self, event):
        pos_in_item = event.pos()
        t_x = pos_in_item.x()
        t_y = pos_in_item.y()
        self.signal_wrapper.hoverMove.emit(self.id, int(pos_in_item.x()), int(pos_in_item.y()))
        self.highlight_area((pos_in_item.x(), pos_in_item.y()))


class ActivationCanvas(QGraphicsView):
    def __init__(self, parent=None, predecessors=None):
        super(ActivationCanvas, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_items = []
        self.highlighted_items = []
        self.predecessors = predecessors

    def add_activation(self, pixmap, pos_x, pos_y, id, pred=[]):
        item = InteractivePixmapItem(pixmap, id=id, predecessors=pred)
        item.signal_wrapper.hoverMove.connect(self.handle_hover_move)
        item.setPos(pos_x, pos_y)
        self.scene.addItem(item)
        self.pixmap_items.append(item)
        return item

    def handle_hover_move(self, item_id, x, y):
        self.remove_highlighting()  # Clear previous highlights

        item = self.find_item_by_id(item_id)
        if item:
            pos = QPointF(x, y)
            item.highlight_area((pos.x(), pos.y()))  # Highlight current item
            t_x = pos.x()
            t_y = pos.y()

        # Propagate highlight to predecessors
        for predecessor_id in self.predecessors.get(item_id, []):
            predecessor_item = self.find_item_by_id(predecessor_id)
            if predecessor_item:
                mapped_pos = self.map_to_predecessor(item_id, predecessor_id, pos)
                if mapped_pos is not None:
                    predecessor_item.highlight_area(mapped_pos)
                if self.predecessors.get(predecessor_id, False):
                    for predecessor_2_id in self.predecessors.get(predecessor_id, []):
                        item_2 = self.find_item_by_id(predecessor_2_id)
                        mapped_pos = self.map_to_predecessor(predecessor_id, predecessor_2_id, mapped_pos)
                        if mapped_pos is not None:
                            item_2.highlight_area(mapped_pos)

    def map_to_predecessor(self, current_id, predecessor_id, pos, B_width=248, A_width=280, A_height=280):
        """
        Maps a position in a current layer (B1 or B2) to the corresponding area in A1.
        """
        if current_id in ['B1', 'B2'] and predecessor_id == 'A1':
            mapped_positions = []
            if not isinstance(pos, list):
                pos = [pos]
            for p in pos:
                if not isinstance(p, tuple):
                    p = (p.x(), p.y())
                kernel_size = 5
                padding = 0  # No padding
                stride = 1  # Stride of the convolution

                # Scale factor calculation needs to account for reduction due to convolution
                scale_x = A_width / B_width
                scale_y = A_height / B_width  # Assuming square dimensions for simplicity

                # Calculate the mapped position in the A image
                scaled_x = int(p[0] * scale_x)
                scaled_y = int(p[1] * scale_y)

                # Calculate the receptive field in A1
                start_x = max(scaled_x - 2, 0)
                start_y = max(scaled_y - 2, 0)
                end_x = min(start_x + kernel_size, A_width)
                end_y = min(start_y + kernel_size, A_height)

                # Generate all coordinates in the receptive field, clamping to A1 dimensions
                mapped_positions += [(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y)]
            return mapped_positions

        elif current_id[0].lower() == "c" and predecessor_id[0].lower() == "b":
            kernel_size = 5
            padding = 0
            stride = 1
            pool_size = 2
            pool_stride = 2

            # Calculate position in the pooled image
            pooled_x = int(pos.x() * (B_width / A_width))
            pooled_y = int(pos.y() * (B_width / A_width))

            # Calculate the receptive field in the pooled image
            start_x = max(pooled_x * pool_stride - 2, 0)
            start_y = max(pooled_y * pool_stride - 2, 0)
            end_x = min(start_x + kernel_size, B_width)
            end_y = min(start_y + kernel_size, B_width)

            # Each position in the receptive field maps to a pool_size x pool_size block in B
            mapped_positions = []
            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    block_start_x = x * pool_size
                    block_start_y = y * pool_size
                    for bx in range(block_start_x, block_start_x + pool_size):
                        for by in range(block_start_y, block_start_y + pool_size):
                            if bx < B_width and by < B_width:
                                mapped_positions.append((bx, by))

            return mapped_positions

        return []

    def find_item_by_id(self, item_id):
        # Find the item with the specified ID in pixmap_items
        for item in self.pixmap_items:
            if item.id == item_id:
                return item
        return None

    def remove_highlighting(self):
        # Remove highlighting from previously highlighted items
        for item in self.highlighted_items:
            item.remove_highlight()
        self.highlighted_items = []


class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 200)
        self.last_point = QPoint()
        self.pen = QPen(Qt.black, 16)
        self.path = QPainterPath()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.white)
        rect = QRect(0, 0, self.width(), self.height())
        painter.setPen(Qt.black)
        painter.drawRect(rect)
        painter.setPen(self.pen)
        painter.drawPath(self.path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()
            self.path.moveTo(self.last_point)
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            new_point = event.pos()
            self.path.lineTo(new_point)
            self.last_point = new_point
            self.update()

    def resetDrawing(self):
        self.path = QPainterPath()
        self.update()

    def getDrawingAsArray(self):
        image = QImage(self.size(), QImage.Format_RGB32)
        image.fill(Qt.white)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(self.pen)
        painter.drawPath(self.path)
        painter.end()

        arr = qimage2ndarray.rgb_view(image)
        grayscale_image = np.array(np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]), dtype=float)
        min_val = np.min(grayscale_image)
        max_val = np.max(grayscale_image)
        grayscale_image_norm = 2 * (grayscale_image - min_val) / (max_val - min_val) - 1
        grayscale_image_norm = np.expand_dims(grayscale_image_norm, axis=-1)
        resized_array = cv2.resize(grayscale_image_norm, (28, 28), interpolation=cv2.INTER_AREA)
        resized_array = np.expand_dims(resized_array, axis=0)  # Add batch dimension
        resized_array = np.expand_dims(resized_array, axis=0)  # Add channel dimension

        input_tensor = torch.tensor(resized_array).float()
        return input_tensor


class ClassificationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.labels = []

        for i in range(10):
            label = QLabel(f"Probability of digit {i}: 0.0")
            layout.addWidget(label)
            self.labels.append(label)
        self.setLayout(layout)

    def updateProbabilities(self, probabilities):
        for i, prob in enumerate(probabilities):
            self.labels[i].setText(f"Probability of {i}: {prob:.2f}")


class TabletWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Activation Visualization")
        self.setGeometry(100, 100, 1000, 850)

        self.drawing_widget = DrawingWidget()
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.drawing_widget.resetDrawing)

        self.activate_button = QPushButton("Activate")
        self.activate_button.clicked.connect(self.activateCNN)

        self.canvas = ActivationCanvas(self, predecessors)

        # Buttons Widget
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.activate_button)
        buttons_widget.setLayout(buttons_layout)

        # left widget
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.drawing_widget)
        left_layout.addWidget(buttons_widget)
        left_widget.setLayout(left_layout)

        # Other Widgets Widget
        other_widgets_widget = QWidget()
        other_widgets_layout = QVBoxLayout()
        other_widgets_layout.addWidget(self.canvas)
        other_widgets_widget.setLayout(other_widgets_layout)

        self.classification_widget = ClassificationWidget()

        self.layout = QHBoxLayout()
        self.layout.addWidget(left_widget)
        self.layout.addWidget(other_widgets_widget)
        self.layout.addWidget(self.classification_widget)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)


    def activateCNN(self):
        try:
            input_tensor = self.drawing_widget.getDrawingAsArray()
            pool, activations, pool1, activations2 = model(input_tensor)
            activations = activations.detach().cpu().numpy()
            pool1 = pool1.detach().cpu().numpy()
            activations2 = activations2.detach().cpu().numpy()
            probabilities = F.softmax(pool, dim=1).detach().cpu().numpy().squeeze()
            self.classification_widget.updateProbabilities(probabilities)

            activation_arrays = [activations[0, i] for i in range(activations.shape[1])]
            pool1_arrays = [pool1[0, i] for i in range(pool1.shape[1])]
            activation2_arrays = [activations2[0, i] for i in range(activations2.shape[1])]
            input_img_array = input_tensor.numpy().squeeze()
            self.displayActivations(input_img_array, activation_arrays, pool1_arrays, activation2_arrays)
        except Exception as e:
            print("An error occurred:", e)

    def displayActivations(self, input_img, activations, pool1, activations2):
        self.canvas.pixmap_items.clear()

        activation2_width = 0
        for activation in activations2:
            activation2_pixmap = array_to_qpixmap(activation, scale=10)
            activation2_width += activation2_pixmap.width() + 10

        activation1_width = 0
        for activation in activations:
            activation_pixmap = array_to_qpixmap(activation, scale=10)
            activation1_width += activation_pixmap.width() + 10

        pool_width = 0
        for pool in pool1:
            pool_pixmap = array_to_qpixmap(pool, scale=10)
            pool_width += pool_pixmap.width() + 10

        total_width = max(pool_width, activation1_width, activation2_width)

        x_offset = 0
        y_offset = 0
        # Convert input image array to QPixmap before using it
        input_pixmap = array_to_qpixmap(input_img, scale=10)
        input_width = input_pixmap.width()
        x_offset = (total_width - input_width) / 2 - 10

        predecessors_in = self.canvas.add_activation(input_pixmap, x_offset, y_offset, "A1")
        # Now use the QPixmap's height for offset calculation
        y_offset += input_pixmap.height() + 10  # Adjust spacing
        x_offset = 0

        i = 1
        predecessors_1 = []
        for activation in activations:
            activation_pixmap = array_to_qpixmap(activation, scale=10)
            item = self.canvas.add_activation(activation_pixmap, x_offset, y_offset, f"B{i}", [predecessors_in])
            predecessors_1.append(item)
            i += 1
            x_offset += activation_pixmap.width() + (
                    total_width - 2 * activation_pixmap.width()) - 10  # Adjust spacing for next activation

        y_offset += activation_pixmap.height() + 10
        x_offset = 0

        i = 1
        for activation in activations2:
            activation2_pixmap = array_to_qpixmap(activation, scale=10)
            _ = self.canvas.add_activation(activation2_pixmap, x_offset, y_offset, f"C{i}", predecessors_1)
            i += 1
            x_offset += activation2_pixmap.width() + 10  # Next activation position

        y_offset += activation2_pixmap.height() + 10
        x_offset = 0


if __name__ == "__main__":
    model = MyConvNet()
    model.load_state_dict(torch.load('mnist-classifier.pt'))
    model.eval()

    app = QApplication(sys.argv)
    window = TabletWindow()
    window.show()
    sys.exit(app.exec_())
