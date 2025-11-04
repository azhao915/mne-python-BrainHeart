from PyQt5.QtCore import Qt

DEFAULT_KEYBINDINGS = {
    Qt.Key_Right: ('time_manager', 'scroll_forward', [], {}),
    Qt.Key_Left: ('time_manager', 'scroll_backward', [], {}),
    Qt.Key_Up: ('channel_manager', '_prev_channel', [], {}),
    Qt.Key_Down: ('channel_manager', '_next_channel', [], {}),
    Qt.Key_Home: ('time_manager', 'zoom_in', [0.8], {}),
    Qt.Key_End: ('time_manager', 'zoom_out', [1.25], {}),
    Qt.Key_Return: ('time_manager', 'to_next_annotation', [], {}),
    Qt.Key_Delete: ('annot_manager', 'toggle_annotations', [], {}),
}

class KeybindingManager: 
    def __init__(self, browser, keybindings = None): 
        self.browser = browser
        self.keybindings = keybindings if keybindings is not None else DEFAULT_KEYBINDINGS
        self.bound_methods = {}

        self.build_bindings()

    def build_bindings(self): 
        for key, binding in self.keybindings.items(): 
            attribute_name, method_name, args, kwargs = binding
            if not hasattr(self.browser, attribute_name): 
                continue
            attribute = getattr(self.browser, attribute_name)
            if not hasattr(attribute, method_name):
                continue
            method = getattr(attribute, method_name)
            self.bound_methods[key] = (method, args, kwargs)
    
    def handle_key_press(self, key): 
        if key in self.bound_methods: 
            method, args, kwargs = self.bound_methods[key]
            method(*args, **kwargs)
            return True
        return False
