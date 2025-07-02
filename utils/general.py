import re


def sanitize_filename(input_string: str) -> str:
    """
    Converts a string to a sanitized version that can be used as a file name
    by replacing invalid characters with underscores (_).

    Args:
        input_string (str): The string to sanitize.

    Returns:
        str: A sanitized string safe for file names.
    """
    # Define the characters to keep: letters, numbers, and safe file name symbols
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', input_string)
    # Optionally replace any leftover whitespace or non-visible characters
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized

def merge_short_strings(strings):
    result = []
    i = 0

    while i < len(strings):
        current = strings[i].strip()  # Remove leading/trailing spaces
        current_tokens = current.split()

        # Keep merging while current string has less than 3 tokens and there's a next string
        while len(current_tokens) < 4 and i + 1 < len(strings):
            i += 1
            next_string = strings[i].strip()
            current += " " + next_string
            current_tokens = current.split()

        result.append(current)
        i += 1  # Move to the next string

    return result