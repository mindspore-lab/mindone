def print_banner(title, banner_width=70):
    """Prints a banner with the given title."""
    title_length = len(title)
    left_padding = (banner_width - title_length) // 2
    right_padding = banner_width - title_length - left_padding - 2

    print("#" * banner_width)
    print(f"#{' ' * left_padding}{title}{' ' * right_padding}#")
    print("#" * banner_width)
