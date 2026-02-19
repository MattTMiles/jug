"""
Backend Legend Widget for JUG GUI.
"""

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QFont
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


class _BackendLegendOriginal(QGraphicsObject):
    """
    Original always-visible legend (kept for easy revert).
    
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


class BackendLegend(QGraphicsObject):
    """
    Collapsible drop-down legend for backends/receivers.

    Shows a small "Backends / Receivers ▼" button in the top-right corner.
    Clicking it toggles a semi-transparent panel listing the backend colour
    swatches (identical to the old always-visible legend).
    """

    _BUTTON_HEIGHT = 22
    _SWATCH_SIZE = 10
    _TEXT_PADDING = 5
    _ROW_HEIGHT = 16
    _PADDING = 8
    _MAX_WIDTH = 180

    def __init__(self, offset=(10, 10), parent=None):
        super().__init__(parent)
        self._offset = offset
        self._items = []          # [(name, QColor), ...]
        self._expanded = False
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(1000)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setAcceptedMouseButtons(Qt.LeftButton)

    # ---- public API (same as original) --------------------------------

    def setItems(self, backend_color_map):
        """Set legend items from backend_name → QColor dict."""
        self._items = sorted(backend_color_map.items())
        self.prepareGeometryChange()
        self.update()

    def setAnchor(self, view_box):
        """Position in the top-right of *view_box*."""
        if not view_box or not view_box.scene():
            return
        vb_rect = view_box.mapRectToScene(view_box.boundingRect())
        my_rect = self.boundingRect()
        x = vb_rect.right() - my_rect.width() - self._offset[0]
        y = vb_rect.top() + self._offset[1]
        self.setPos(x, y)

    # ---- geometry / painting ------------------------------------------

    def _button_rect(self):
        return QRectF(0, 0, self._MAX_WIDTH, self._BUTTON_HEIGHT)

    def _panel_rect(self):
        if not self._items:
            return QRectF()
        n = len(self._items)
        h = self._PADDING * 2 + n * self._ROW_HEIGHT
        return QRectF(0, self._BUTTON_HEIGHT, self._MAX_WIDTH, h)

    def boundingRect(self):
        r = self._button_rect()
        if self._expanded and self._items:
            r = r.united(self._panel_rect())
        return r

    def paint(self, painter, option, widget):
        from jug.gui.theme import is_dark_mode
        dark = is_dark_mode()

        if dark:
            bg = QColor(26, 19, 37, 200)
            text_col = QColor(246, 238, 228)
        else:
            bg = QColor(253, 252, 250, 220)
            text_col = QColor(32, 30, 27)

        # --- button bar ---
        btn = self._button_rect()
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.NoPen)
        if self._expanded:
            painter.drawRoundedRect(
                QRectF(btn.x(), btn.y(), btn.width(), btn.height() + 4), 4, 4
            )
        else:
            painter.drawRoundedRect(btn, 4, 4)

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QPen(text_col))

        arrow = "▲" if self._expanded else "▼"
        label = f"Backends / Receivers {arrow}"
        painter.drawText(btn.adjusted(self._PADDING, 0, -self._PADDING, 0),
                         Qt.AlignLeft | Qt.AlignVCenter, label)

        if not self._expanded or not self._items:
            return

        # --- drop-down panel ---
        panel = self._panel_rect()
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.NoPen)
        # Draw panel connecting to button (overlap 4px for seamless join)
        painter.drawRoundedRect(
            QRectF(panel.x(), panel.y() - 4, panel.width(), panel.height() + 4),
            4, 4,
        )

        painter.setPen(QPen(text_col))
        y = panel.y() + self._PADDING
        for name, color in self._items:
            swatch = QRectF(
                self._PADDING,
                y + (self._ROW_HEIGHT - self._SWATCH_SIZE) / 2,
                self._SWATCH_SIZE,
                self._SWATCH_SIZE,
            )
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(swatch, 2, 2)

            text_x = self._PADDING + self._SWATCH_SIZE + self._TEXT_PADDING
            text_rect = QRectF(
                text_x, y,
                self._MAX_WIDTH - text_x - self._PADDING,
                self._ROW_HEIGHT,
            )
            painter.setPen(QPen(text_col))
            display = name if len(name) <= 20 else name[:17] + "..."
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, display)
            y += self._ROW_HEIGHT

    # ---- interaction --------------------------------------------------

    def mousePressEvent(self, event):
        if self._button_rect().contains(event.pos()):
            self.prepareGeometryChange()
            self._expanded = not self._expanded
            self.update()
            event.accept()
        else:
            super().mousePressEvent(event)
