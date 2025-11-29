import json
import os

# cwd_path is a relative path from the current working directory (root of the project)

def get_cwd_root_path():
    """
    Returns the current working directory path.
    """
    #return os.getcwd()
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_or_make_path(full_path):
    """
    Creates a directory at the specified path if it does not already exist.
    """
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

async def aget_or_make_path(full_path):
    """
    Asynchronously creates a directory at the specified path if it does not already exist.
    """
    import aiofiles.os
    if not await aiofiles.os.path.exists(full_path):
        await aiofiles.os.makedirs(full_path)
    return full_path

def check_path_exists(full_path):
    """
    Checks if a cwd path exists at the specified path.
    """
    return os.path.exists(full_path)

def get_all_file_under(base_path : str, file_name : str):
    index_data = []
    root_path = get_cwd_root_path()
    full_base_path = os.path.join(root_path, base_path)

    for root, dirs, files in os.walk(full_base_path):
        if file_name in files:
            index_path = os.path.join(root, "index.json")
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    data = json.loads(content)
                    index_data.append(data)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decoding failed for {index_path}: {e}")
            except Exception as e:
                print(f"[ERROR] Failed to read {index_path}: {e}")

    return index_data

def LOG_TEXT(message, colorKey="", verbosity=""):
    """Simple logging function with color support. Use sparingly."""
    if verbosity == "DEBUG":
        return

    COLORMAP = {
        'BLUE': '\033[94m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'MAGENTA': '\033[95m',
    }

    RESET = '\033[0m'
    color = COLORMAP.get(colorKey, RESET)

    if verbosity and verbosity != "INFO":
        print(f"{color}{verbosity}: {message}{RESET}")
    else:
        print(f"{color}{message}{RESET}")