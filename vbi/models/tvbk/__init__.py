

try:
    import tvbk as m
    from .tvbk_wrapper import MPR
    TVBK_AVAILABLE = True
except ImportError:
    TVBK_AVAILABLE = False
    print("Tvbtk not available. Please install it.")
