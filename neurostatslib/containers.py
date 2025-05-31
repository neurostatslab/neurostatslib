# from hdmf.container import Container
# from hdmf.utils import get_basic_array_info, generate_array_html_repr
import numpy as np
from collections import UserDict
import scipy


def _get_terminal_size():
    """Helper to get terminal size for __repr__

    Returns
    -------
    tuple

    """
    cols = 100  # Default
    rows = 2
    try:
        cols, rows = os.get_terminal_size()
    except Exception:
        import shutil

        cols, rows = shutil.get_terminal_size()

    return (cols, rows)


def get_basic_array_info(array):
    def convert_bytes_to_str(bytes_size):
        suffixes = ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB"]
        i = 0
        while bytes_size >= 1024 and i < len(suffixes) - 1:
            bytes_size /= 1024.0
            i += 1
        return f"{bytes_size:.2f} {suffixes[i]}"

    if hasattr(
        array, "nbytes"
    ):  # TODO: Remove this after h5py minimal version is larger than 3.0
        array_size_in_bytes = array.nbytes
    else:
        array_size_in_bytes = array.size * array.dtype.itemsize
    array_size_repr = convert_bytes_to_str(array_size_in_bytes)
    basic_array_info_dict = {
        "Data type": array.dtype,
        "Shape": array.shape,
        "Array size": array_size_repr,
    }

    return basic_array_info_dict


def generate_array_html_repr(array_info_dict, array, dataset_type=None):
    def html_table(item_dicts) -> str:
        """
        Generates an html table from a dictionary
        """
        report = '<table class="data-info">'
        report += "<tbody>"
        for k, v in item_dicts.items():
            report += (
                f"<tr>"
                f'<th style="text-align: left">{k}</th>'
                f'<td style="text-align: left">{v}</td>'
                f"</tr>"
            )
        report += "</tbody>"
        report += "</table>"
        return report

    array_info_html = html_table(array_info_dict)
    repr_html = (
        dataset_type + "<br>" + array_info_html
        if dataset_type is not None
        else array_info_html
    )

    # Array like might lack nbytes (h5py < 3.0) or size (DataIO object)
    if hasattr(array, "nbytes"):
        array_size_bytes = array.nbytes
    else:
        if hasattr(array, "size"):
            array_size = array.size
        else:
            import math

            array_size = math.prod(array.shape)
        array_size_bytes = array_size * array.dtype.itemsize

    # Heuristic for displaying data
    array_is_small = (
        array_size_bytes < 1024 * 0.1
    )  # 10 % a kilobyte to display the array
    if array_is_small:
        repr_html += "<br>" + str(np.asarray(array))

    return repr_html


def get_container_list_info(containers):
    container_list_info = {"index": [], "fields": []}
    for container in containers:
        container_list_info["index"].append(container.name)
        container_list_info["fields"].append("\n".join(list(container.data.keys())))

    return container_list_info


def generate_list_html_repr(container_info_dict, dataset_type=None):
    def html_table(item_dicts) -> str:
        """
        Generates an html table from a dictionary
        """
        report = '<table class="data-info">'
        report += "<tbody>"
        for k, v in item_dicts.items():
            report += f"<tr>" f'<th style="text-align: left">{k}</th>'
            for i in v:
                report += f'<td style="text-align: left">{i}</td>'
            report += f"</tr>"
        report += "</tbody>"
        report += "</table>"
        return report

    array_info_html = html_table(container_info_dict)
    repr_html = (
        dataset_type + "<br>" + array_info_html
        if dataset_type is not None
        else array_info_html
    )

    return repr_html


class MatStruct(UserDict):
    """A container that can contain other containers and has special functionality for printing."""

    def __init__(self, data=None, name="root"):

        super().__init__(data)
        self.name = name

    @property
    def fields(self):
        return self.data

    def __getattr__(self, key):
        if key in self.data:
            if isinstance(self.data[key], dict):
                return MatStruct(self.data[key], key)
            elif isinstance(self.data[key], list) and isinstance(
                self.data[key][0], dict
            ):
                return MatCell(self.data[key], name=key)
                # return [Container(d, key+"["+str(k)+"]") for k,d in enumerate(self.data[key])]

            else:
                return self.data[key]
        else:
            super().__getattr__(key)

    def __repr__(self):
        return MatStruct.__smart_str_struct(self, 0)

    @property
    def css_style(self) -> str:
        """CSS styles for the HTML representation."""
        return """
        <style>
            .container-fields {
                font-family: "Open Sans", Arial, sans-serif;
            }
            .container-fields .field-value {
                color: #00788E;
            }
            .container-fields details > summary {
                cursor: pointer;
                display: list-item;
            }
            .container-fields details > summary:hover {
                color: #0A6EAA;
            }
        </style>
        """

    @property
    def js_script(self) -> str:
        """JavaScript for the HTML representation."""
        return """
        <script>
            function copyToClipboard(text) {
                navigator.clipboard.writeText(text).then(function() {
                    console.log('Copied to clipboard: ' + text);
                }, function(err) {
                    console.error('Could not copy text: ', err);
                });
            }

            document.addEventListener('DOMContentLoaded', function() {
                let fieldKeys = document.querySelectorAll('.container-fields .field-key');
                fieldKeys.forEach(function(fieldKey) {
                    fieldKey.addEventListener('click', function() {
                        let accessCode = fieldKey.getAttribute('title').replace('Access code: ', '');
                        copyToClipboard(accessCode);
                    });
                });
            });
        </script>
        """

    def _repr_html_(self) -> str:
        """Generates the HTML representation of the object."""
        header_text = (
            self.name
            if self.name == self.__class__.__name__
            else f"{self.name} ({self.__class__.__name__})"
        )
        html_repr = self.css_style + self.js_script
        html_repr += "<div class='container-wrap'>"
        html_repr += f"<div class='container-header'><div class='xr-obj-type'><h3>{header_text}</h3></div></div>"
        html_repr += self._generate_html_repr(self.fields, is_field=True)
        html_repr += "</div>"
        return html_repr

    def _generate_html_repr(self, fields, level=0, access_code="", is_field=False):
        """Recursively generates HTML representation for fields."""
        html_repr = ""

        if isinstance(fields, dict):
            for key, value in fields.items():
                current_access_code = (
                    f"{access_code}.{key}" if is_field else f"{access_code}['{key}']"
                )
                if hasattr(value, "_generate_field_html"):
                    html_repr += value._generate_field_html(
                        key, value, level, current_access_code
                    )
                else:
                    html_repr += self._generate_field_html(
                        key, value, level, current_access_code
                    )
        elif isinstance(fields, list):
            if len(fields) and isinstance(fields[0], dict):
                items = MatCell(fields, name=self.name)
                container_list_info = get_container_list_info(items.containers)
                html_repr += generate_list_html_repr(container_list_info, "MatCell")
            else:
                for index, item in enumerate(fields[:10]):
                    access_code += f"[{index}]"
                    html_repr += self._generate_field_html(
                        index, item, level, access_code
                    )
        else:
            pass

        return html_repr

    def _generate_field_html(self, key, value, level, access_code):
        """Generates HTML for a single field.

        This function can be overwritten by a child class to implement customized html representations.
        """

        if isinstance(value, (int, float, str, bool)):
            return (
                f'<div style="margin-left: {level * 20}px;" class="container-fields"><span class="field-key"'
                f' title="{access_code}">{key}: </span><span class="field-value">{value}</span></div>'
            )

        # Detects array-like objects that conform to the Array Interface specification
        # (e.g., NumPy arrays, HDF5 datasets, DataIO objects). Objects must have both
        # 'shape' and 'dtype' attributes. Iterators are excluded as they lack 'shape'.
        # This approach keeps the implementation generic without coupling to specific backends methods
        is_array_data = hasattr(value, "shape") and hasattr(value, "dtype")

        if is_array_data:
            html_content = self._generate_array_html(value, level + 1)
        elif isinstance(value, (list, dict, np.ndarray)):
            html_content = self._generate_html_repr(
                value, level + 1, access_code, is_field=False
            )
        else:
            html_content = f'<span class="field-key">{value}</span>'

        html_repr = (
            f'<details><summary style="display: list-item; margin-left: {level * 20}px;" '
            f'class="container-fields field-key" title="{access_code}"><b>{key}</b></summary>'
        )
        html_repr += html_content
        html_repr += "</details>"

        return html_repr

    def _generate_array_html(self, array, level):
        """Generates HTML for array data (e.g., NumPy arrays, HDF5 datasets, Zarr datasets and DataIO objects)."""

        is_numpy_array = isinstance(array, np.ndarray)
        # read_io = self.get_read_io()
        # it_was_read_with_io = read_io is not None
        # is_data_io = isinstance(array, DataIO)

        if is_numpy_array:
            array_info_dict = get_basic_array_info(array)
            repr_html = generate_array_html_repr(array_info_dict, array, "NumPy array")
        else:  # Not sure which object could get here
            object_class = array.__class__.__name__
            array_info_dict = get_basic_array_info(array.data)
            repr_html = generate_array_html_repr(
                array_info_dict, array.data, object_class
            )

        return f'<div style="margin-left: {level * 20}px;" class="container-fields">{repr_html}</div>'

    @staticmethod
    def _smart_str(v, num_indent):
        """
        Print compact string representation of data.

        If v is a list, try to print it using numpy. This will condense the string
        representation of datasets with many elements. If that doesn't work, just print the list.

        If v is a dictionary, print the name and type of each element

        If v is a set, print it sorted

        If v is a neurodata_type, print the name of type

        Otherwise, use the built-in str()
        Parameters
        ----------
        v

        Returns
        -------
        str

        """

        if isinstance(v, list) or isinstance(v, tuple):
            if len(v) and isinstance(v[0], MatStruct):
                return MatStruct.__smart_str_list(v, num_indent, "(")
            try:
                return str(np.asarray(v))
            except ValueError:
                return MatStruct.__smart_str_list(v, num_indent, "(")
        elif isinstance(v, dict):
            return MatStruct.__smart_str_dict(v, num_indent)
        elif isinstance(v, set):
            return MatStruct.__smart_str_list(sorted(list(v)), num_indent, "{")
        elif isinstance(v, MatStruct):
            return MatStruct.__smart_str_struct(v, num_indent)
        else:
            return v.__repr__() if hasattr(v, "__repr__") else str(v)

    @staticmethod
    def __smart_str_struct(struct, num_indent):
        # cls = struct.__class__
        # template = "%s %s.%s" % (struct.name, cls.__module__, cls.__name__)
        template = "%s %s" % (struct.name, type(struct))
        if len(struct.fields):
            template += "\n Fields:\n"
        for k in sorted(struct.fields):  # sorted to enable tests
            v = struct.fields[k]
            if hasattr(v, "__len__"):
                # if isinstance(v, (np.ndarray, list, tuple)) or v:
                template += " " * num_indent + "  {}: {}\n".format(
                    k, MatStruct._smart_str(v, num_indent + 1)
                )
            else:
                template += " " * num_indent + "  {}: {}\n".format(k, v)
        return template

    @staticmethod
    def __smart_str_list(str_list, num_indent, left_br):
        if left_br == "(":
            right_br = ")"
        if left_br == "{":
            right_br = "}"
        if len(str_list) == 0:
            return left_br + " " + right_br
        indent = num_indent * 2 * " "
        indent_in = (num_indent + 1) * 2 * " "
        out = left_br
        for v in str_list[:-1]:
            out += "\n" + indent_in + MatStruct._smart_str(v, num_indent + 1) + ","
        if str_list:
            out += "\n" + indent_in + MatStruct._smart_str(str_list[-1], num_indent + 1)
        out += "\n" + indent + right_br
        return out

    @staticmethod
    def __smart_str_dict(d, num_indent):
        left_br = "{"
        right_br = "}"
        if len(d) == 0:
            return left_br + " " + right_br
        indent = num_indent * 2 * " "
        indent_in = (num_indent + 1) * 2 * " "
        out = left_br
        keys = sorted(list(d.keys()))
        for k in keys[:-1]:
            # out += '\n' + indent_in + MatStruct._smart_str(k, num_indent + 1) + ' ' + str(type(d[k])) + ','
            out += (
                "\n"
                + indent_in
                + MatStruct._smart_str(k, num_indent + 1)
                + ": "
                + MatStruct._smart_str(d[k], 1)
                + ","
            )

        if keys:
            # out += '\n' + indent_in + MatStruct._smart_str(keys[-1], num_indent + 1) + ' ' + str(type(d[keys[-1]]))
            out += (
                "\n"
                + indent_in
                + MatStruct._smart_str(keys[-1], num_indent + 1)
                + ": "
                + MatStruct._smart_str(d[keys[-1]], 1)
            )

        out += "\n" + indent + right_br
        return out


class MatCell:
    def __init__(self, data, name="root"):

        self.name = name
        self.containers = [
            MatStruct(d, name + "[" + str(k) + "]") for k, d in enumerate(data)
        ]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.containers[key]
        else:
            super().__getitem__(key)

    def __repr__(self):
        cls = self.__class__
        template = "%s %s.%s with length %d" % (
            self.name,
            cls.__module__,
            cls.__name__,
            len(self.containers),
        )
        # if len(self.fields):
        #     template += "\nFields:\n"
        _, nrows = _get_terminal_size()
        k = 0
        while template.count("\n") < nrows:
            template += "\n" + MatStruct._smart_str(self.containers[k], 2)
            k += 1
        if k < len(self.containers):
            template = template.split("\n")[: nrows - 2]
            template = "\n ".join(template)
            temp_end = "\n" + MatStruct._smart_str(self.containers[-1], 2)
            temp_end = temp_end.split("\n")[-4:-2]  # last 2 lines should be empty
            template += "\n...\n" + "\n ".join(temp_end)

        return template

    def _repr_html_(self) -> str:
        """Generates the HTML representation of the object."""
        header_text = (
            self.name
            if self.name == self.__class__.__name__
            else f"{self.name} ({self.__class__.__name__})"
        )
        html_repr = "<div class='container-wrap'>"
        html_repr += f"<div class='container-header'><div class='xr-obj-type'><h3>{header_text}</h3></div></div>"
        container_list_info = get_container_list_info(self.containers)
        html_repr += generate_list_html_repr(container_list_info)
        html_repr += "</div>"
        return html_repr


# recursive function to create nested containers
def get_type(k, d):
    if isinstance(d, dict):
        return mat_struct(k, d)
    elif isinstance(d, (np.ndarray, list)):
        d = np.array(d)
        try:
            d = np.stack(*d)
        except:
            pass
        # if np.issubdtype(d.dtype, object):
        #     return MatCell(k, d)
        if len(d.shape):
            if np.issubdtype(d.dtype, object):
                # return MatCell(k, d)
                return mat_cell(k, d)
            # return MatArray(k, d)
            else:
                return d
        elif np.issubdtype(d.dtype, np.number):
            return float(d)  # MatField(k, float(d))
        else:
            return str(d)  # MatField(k, str(d))
    else:
        return d  # MatField(k, d)


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

    FLDS = list(struct.keys())
    if len(FLDS) > 1:
        # HDMF container extended with fields set by keys of the input
        class MatFile(MatStruct):
            __fields__ = tuple(FLDS)

        container = MatFile(name=name)
        # set fields of the container
        for parent_key, parent_data in struct.items():
            setattr(container, parent_key, get_type(parent_key, parent_data))
    else:
        # if the input is a single field, we can just return the data
        # this is useful for loading a single file
        container = get_type(FLDS[0], struct[FLDS[0]])

    return container
