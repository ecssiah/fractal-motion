def print_percentage(current: float, total: float, label: str) -> None:
    percent = current / total * 100

    print(f'\r{label: <10}{percent:.1f}% ', end='', flush=True)


def time_string(elapsed_seconds: float) -> str:
    hours, seconds = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"


def to_superscript(number) -> str:
    superscript_map = str.maketrans(
        '0123456789', 
        '⁰¹²³⁴⁵⁶⁷⁸⁹',
    )

    return str(number).translate(superscript_map)