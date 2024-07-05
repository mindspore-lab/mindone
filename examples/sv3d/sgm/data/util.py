def _is_valid_text_input(t):
    if isinstance(t, str):
        # Strings are fine
        return True
    elif isinstance(t, (list, tuple)):
        # List are fine as long as they are...
        if len(t) == 0:
            # ... empty
            return True
        elif isinstance(t[0], str):
            # ... list of strings
            return True
        elif isinstance(t[0], (list, tuple)):
            # ... list with an empty list or with a list of strings
            return len(t[0]) == 0 or isinstance(t[0][0], str)
        else:
            return False
    else:
        return False
