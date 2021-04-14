import pickle
from functools import wraps
import os

def pickle_memoize(make_filename):
    """Decorator which memoizes the results of function calls with pickle

    Saves the result of calling the function to make_filename(*args,**kwargs)

    If this path already exists, then the contents are unpickled rather than
    calling the function.

    Safe to call concurrently across multiple threads/processes.
    """

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            filename = make_filename(*args, **kwargs)
            try:
                with open(filename, "rb") as pickle_file:
                    return pickle.load(pickle_file)
            except FileNotFoundError:
                result = func(*args, **kwargs)
                try:
                    with open(filename, "xb") as pickle_file:
                        pickle.dump(result, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
                except FileExistsError:
                    pass
                except:
                    os.remove(filename)
                    raise

                return result
        return wrapped
    return decorator
