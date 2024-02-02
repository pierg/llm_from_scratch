from pathlib import Path


def read_words(filename: Path) -> list:
    """
    Reads words from a file.

    :param filename: A path to the file.
    :return: A list of words.
    """
    with open(filename, "r") as file:
        words = file.read().splitlines()
    return words


def create_mappings(words: list[str]) -> tuple:
    """
    Creates mappings from characters to indices and vice versa.

    :return: A tuple of two dictionaries (stoi, itos).
    """
    unique_chars = sorted(list(set("".join(words))))
    stoi = {s: enum+1 for enum, s in enumerate(unique_chars)}
    stoi["."] = 0
    stoi = dict(sorted([(k, v) for k, v in stoi.items()], key=lambda x: x[1]))
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


