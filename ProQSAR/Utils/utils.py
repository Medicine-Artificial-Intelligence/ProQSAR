import json
from typing import List, Dict


def save_database(database: list[dict], pathname: str = "./Data/database.json") -> None:
    """
    Serializes a database, represented as a list of dictionaries, to a JSON file.

    Parameters:
    - database (list[dict]): The database to be serialized and saved.
    - pathname (str): The file path to save the JSON representation of the database.
      Defaults to './Data/database.json'.

    Raises:
    - TypeError: If the input `database` is not a list of dictionaries.
    - ValueError: If an error occurs during file writing, capturing specifics of the IO error.

    Examples:
    - save_database([{'name': 'Alice', 'age': 30}], './users.json')  # Saves the data to 'users.json'
    """
    if not all(isinstance(item, dict) for item in database):
        raise TypeError("Database should be a list of dictionaries.")

    try:
        with open(pathname, "w") as f:
            json.dump(database, f)
    except IOError as e:
        raise ValueError(f"Error writing to file {pathname}: {e}")


def load_database(pathname: str = "./Data/database.json") -> List[Dict]:
    """
    Deserializes a JSON file into a database, which is returned as a list of dictionaries.

    Parameters:
    - pathname (str): The file path from which the database will be loaded.
      Defaults to './Data/database.json'.

    Returns:
    - list[dict]: The deserialized list of dictionaries representing the database.

    Raises:
    - ValueError: If an error occurs during file reading, with details about the IO error.

    Examples:
    - database = load_database('./users.json')  # Loads data from 'users.json'
    """
    try:
        with open(pathname, "r") as f:
            database = json.load(f)  # Load the JSON data from the file
        return database
    except IOError as e:
        raise ValueError(f"Error reading from file {pathname}: {e}")
