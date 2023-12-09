class BokehUi:
    """
    Helper class to compose Bokeh UI elements into complex classes.

    Currently it is NOT possible to compose widgets (e.g., via sub-classing). Instead,
    use regular python with a single method ``get_ui`` that returns the Bokeh element to y_axis
    of the complex widget.
    """
    def __init__(self, ui):
        self.ui = ui
        assert ui is not None

    def get_ui(self):
        return self.ui