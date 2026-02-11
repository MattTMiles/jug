"""
Simple Horizontal Color Bar for JUG GUI.
"""

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QLinearGradient, QColor, QBrush, QPen, QFont
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem

class SimpleColorBar(QGraphicsObject):
    """
    A minimal, horizontal color bar with min/max labels.
    
    Draws a gradient rectangle and text labels. 
    Designed to overlay on a PlotItem.
    """
    
    def __init__(self, size=(120, 12), offset=(10, 10), parent=None):
        super().__init__(parent)
        self._size = size  # (width, height) used for the gradient strip
        self._offset = offset # (x, y) margin from top-right corner
        self._gradient_stops = []
        self._labels = {"min": "", "max": ""}
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations) # Keep size constant on zoom
        self.setZValue(1000) # Ensure it renders above plot items
        
        # Performance: Cache the gradient painting to avoid X11 flooding
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        
        # Default gradient (Purple -> Red -> Yellow)
        self.setGradient([
            (0.0, (68, 1, 84, 255)),
            (0.5, (193, 60, 100, 255)),
            (1.0, (253, 231, 37, 255))
        ])

    def setGradient(self, stops):
        """Set gradient stops: [(pos, (r,g,b,a)), ...]"""
        self._gradient_stops = stops
        self.update()

    def setLabels(self, min_text, max_text):
        """Set min/max labels."""
        self._labels["min"] = min_text
        self._labels["max"] = max_text
        self.update()

    def boundingRect(self):
        # Return bounding box covering gradient + labels
        # Gradient is (0,0) to (w, h)
        # Text is below or aside.
        # Let's keep it simple: Text inside or just below?
        # User said "unassuming horizontal". 
        # Layout: [Min] [--- Gradient ---] [Max]
        # Total width = 2*text_width + gradient_width + spacing
        # Or: 
        # [--- Gradient ---]
        # min              max
        
        # Let's do: [Gradient Strip] with text below ends.
        w, h = self._size
        return QRectF(0, 0, w, h + 15) # +15 for text

    def paint(self, painter, option, widget):
        w, h = self._size
        
        # Draw Gradient
        grad = QLinearGradient(0, 0, w, 0) # Horizontal
        for pos, color in self._gradient_stops:
            if len(color) == 4:
                c = QColor(*color)
            else:
                c = QColor(color[0], color[1], color[2])
            grad.setColorAt(pos, c)
            
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, 2, 2)
        
        # Draw Labels
        painter.setPen(QPen(QColor(200, 200, 200))) # Light grey text
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        
        # Min label (left aligned below)
        text_rect_min = QRectF(0, h + 2, w/2, 12)
        painter.drawText(text_rect_min, Qt.AlignLeft | Qt.AlignTop, self._labels["min"])
        
        # Max label (right aligned below)
        text_rect_max = QRectF(w/2, h + 2, w/2, 12)
        painter.drawText(text_rect_max, Qt.AlignRight | Qt.AlignTop, self._labels["max"])

    def setAnchor(self, view_box):
        """Position this item relative to a ViewBox (top-right)."""
        if not view_box or not view_box.scene():
            return
            
        # Get view geometry in scene coordinates
        vb_rect = view_box.mapRectToScene(view_box.boundingRect())
        
        # My size (assuming I am in the scene or a parent with identity transform)
        my_rect = self.boundingRect()
        
        # Position at top-right with offset
        # x = right - my_width - offset_x
        # y = top + offset_y
        x = vb_rect.right() - my_rect.width() - self._offset[0]
        y = vb_rect.top() + self._offset[1]
        
        self.setPos(x, y)
