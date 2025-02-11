from hdmf.container import Container
import numpy as np


class MatData(Container):
    """
    HDMF container extended to display properties of array-like data.

    Parameters
    ----------
    name : str
        Name of the data array
    values : np.ndarray
        Values of the data array. Must be a numpy array.

    Attributes
    ----------
    shape : str
        Shape of the data array
    dtype : str
        Data type of the array elements
    values : np.ndarray
        Values of the data array
    """

    __fields__ = (
        "shape",
        "dtype",
        "values",
    )

    def __init__(self, name, values):
        super().__init__(name)
        self.shape = str(np.shape(values))
        # endians don't print for some reason
        self.dtype = str(values.dtype).replace("<", "").replace(">", "")
        self.values = values


class MatField(Container):
    """
    HDMF container extended to display properties of single-valued fields.

    Parameters
    ----------
    name : str
        Name of the field
    value : any
        Value of the field

    Attributes
    ----------
    type : str
        Type of the field
    value : any
        Value of the field
    """

    __fields__ = (
        "type",
        "value",
    )

    def __init__(self, name, value):
        super().__init__(name)
        # endians don't print for some reason
        self.type = type(value).__name__.replace("<", "").replace(">", "")
        self.value = value


def mat_container(struct, name="root"):
    """
    Function to create a container with dynamic fields according to an input dictionary / loaded matlab struct.

    This function is called recursively such that nested structs / dictionaries are represented as nested containers. Arrays are represented as MatData containers, and single values are represented as MatField containers.

    Parameters
    ----------
    struct : dict
        Dictionary or matlab struct loaded with mat73.loadmat or scipy.io.loadmat
    name : str
        Name of the container. On the first call, this will be the name of the outermost container. On nested calls, it will be the dictionary key associated with the conatainer values.
    """

    # recursive function to create nested containers
    def get_container(k, d):
        if isinstance(d, dict):
            return mat_container(d, k)
        elif isinstance(d, (np.ndarray, list)):
            d = np.array(d)
            if len(d.shape):
                return MatData(k, np.array(d))
            elif np.issubdtype(d.dtype, np.number):
                return MatField(k, float(d))
            else:
                return MatField(k, str(d))
        else:
            return MatField(k, d)

    FLDS = struct.keys()

    # HDMF container extended with fields set by keys of the input
    class MatFile(Container):
        __fields__ = tuple(FLDS)

    container = MatFile(name=name)

    # set fields of the container
    for parent_key, parent_data in struct.items():
        setattr(container, parent_key, get_container(parent_key, parent_data))

    return container
