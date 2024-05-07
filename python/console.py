_CONSOLE_COLOR = {
    'blue': '\033[94m',
    'gray': '\033[90m',
    'grey': '\033[90m',
    'yellow': '\033[93m',
    'orange': '\033[93m',  # yellow and orange are actually the same
    'black': '\033[90m',
    'cyan': '\033[96m',
    'green': '\033[92m',
    'magenta': '\033[95m',
    'white': '\033[97m',
    'red': '\033[91m',
    'default': '\033[0m',
}


def color_print(*args, color, **kwargs):
    print(_CONSOLE_COLOR.get(color.lower(), _CONSOLE_COLOR['default']), end="")
    print(*args, **kwargs)
    print(_CONSOLE_COLOR['default'], end="")

