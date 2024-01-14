def print_percentage(current: float, total: float, label: str) -> None:
    percent = current / total * 100

    print(f'\r{label: <10}{percent:.1f}% ', end='', flush=True)