def _init():
    global _glo_dict
    _glo_dict = {}

def set_value(k, v):
    _glo_dict[k] = v

def get_value(k):
    try:
        return _glo_dict[k]
    except:
        return None