"""
Backend Legend Widget for JUG GUI.
"""

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QFont
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


class BackendLegend(QGraphicsObject):
    """
    Compact legend showing backend names with colored swatches.
    
    Designed to overlay on a PlotItem in the top-right corner.
    """
    
    def __init__(self, offset=(10, 10), parent=None):
        super().__init__(parent)
        self._offset = offset  # (x, y) margin from top-right corner
        self._items = []  # List of (backend_name, QColor) tuples
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(1000)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        
        # Layout parameters
        self._swatch_size = 10  # Square color swatch
        self._text_padding = 5  # Space between swatch and text
        self._row_height = 16  # Height of each row
        self._padding = 8  # Internal padding
        self._max_width = 180  # Maximum width before wrapping
    
    def setItems(self, backend_color_map):
        """Set legend items from backend_name → QColor dict.
        
        Parameters
        ----------
        backend_color_map : dict of str → QColor
            Mapping of backend names to colors.
        """
        # Sort alphabetically for stable display
        self._items = sorted(backend_color_map.items())
        self.update()
    
    def boundingRect(self):
        """Calculate bounding rectangle for all items."""
        if not self._items:
            return QRectF(0, 0, 0, 0)
        
        # Calculate layout
        n_items = len(self._items)
        height = self._padding * 2 + n_items * self._row_height
        width = self._max_width
        
        return QRectF(0, 0, width, height)
    
    def paint(self, painter, option, widget):
        """Draw the legend."""
        if not self._items:
            return
        
        rect = self.boundingRect()
        
        # Draw semi-transparent background
        from jug.gui.theme import is_dark_mode
        if is_dark_mode():
            bg_color = QColor(26, 19, 37, 200)  # Dark purple with alpha
            text_color = QColor(246, 238, 228)  # Light text
        else:
            bg_color = QColor(253, 252, 250, 220)  # Warm white with alpha
            text_color = QColor(32, 30, 27)  # Dark text
        
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 4, 4)
        
        # Draw items
        painter.setPen(QPen(text_color))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        
        y = self._padding
        for backend_name, color in self._items:
            # Draw color swatch
            swatch_rect = QRectF(
                self._padding,
                y + (self._row_height - self._swatch_size) / 2,
                self._swatch_size,
                self._swatch_size
            )
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(swatch_rect, 2, 2)
            
            # Draw text
            text_x = self._padding + self._swatch_size + self._text_padding
            text_rect = QRectF(
                text_x,
                y,
                self._max_width - text_x - self._padding,
                self._row_height
            )
            painter.setPen(QPen(text_color))
            
            # Truncate long backend names
            display_name = backend_name
            if len(display_name) > 20:
                display_name = display_name[:17] + "..."
            
            painter.drawText(
                text_rect,
                Qt.AlignLeft | Qt.AlignVCenter,
                display_name
            )
            
            y += self._row_height
    
    def setAnchor(self, view_box):
        """Position this item relative to a ViewBox (top-right)."""
        if not view_box or not view_box.scene():
            return
        
        # Get view geometry in scene coordinates
        vb_rect = view_box.mapRectToScene(view_box.boundingRect())
        
        # My size
        my_rect = self.boundingRect()
        
        # Position at top-right with offset
        x = vb_rect.right() - my_rect.width() - self._offset[0]
        y = vb_rect.top() + self._offset[1]
        
        self.setPos(x, y)
