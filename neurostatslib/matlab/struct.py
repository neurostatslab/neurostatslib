import numpy as np
from typing import Union


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


class MatStruct:
    """
    Class to represent MATLAB structures in Python.
    This class mimics the behavior of MATLAB structs, allowing access to fields
    as attributes and providing a structured representation of the data.

    An HTML representation is also provided for easy visualization in Jupyter notebooks,
    based on styling from NWB objects.

    Parameters
    ----------
    data : dict or np.ndarray
        The data to be represented as a MATLAB struct. It can be a dictionary or a structured NumPy array.
    name : str, optional
        The name of the struct, by default "root". This is used in the HTML representation.

    Attributes
    ----------
    data : np.void or np.ndarray
        The structured data represented as a NumPy void array or structured array.
    name : str
        The name of the struct, used in the HTML representation.
    shape : tuple
        The shape of the struct.
    fields : list
        The fields names of the MATLAB struct, which can be accessed as attributes.
    """

    def __init__(self, data: Union[np.array, np.void, dict], name: str = "root"):
        self.name = name
        if isinstance(data, dict):
            # Convert dict to structured array
            dtype = [(k, "O") for k in data.keys()]
            data = np.void(tuple(data.values()), dtype=dtype)
        if isinstance(data, (np.ndarray, np.void)) and (data.dtype.names is not None):
            self.data = data
        else:
            raise ValueError("Data must be a structured array with named fields.")

    @property
    def shape(self):
        return self.data.shape

    @property
    def fields(self):
        return list(self.data.dtype.names)

    def __dir__(self):
        return ["shape", "fields"] + self.fields

    def __getattr__(self, key: str):
        """
        Access fields as attributes. If the field is not found, it raises an AttributeError.
        If the field is a structured array, it returns a MatStruct object for that field.
        """
        if key in self.fields:
            data = self.data[key]
            if isinstance(data, np.ndarray):
                # If the data is an array, we stack it to ensure it's a single array
                # this will also combine an object array of structured arrays into a single structured array
                data = np.stack(data).squeeze()
            if isinstance(data, (np.ndarray, np.void)) and (
                data.dtype.names is not None
            ):
                data = MatStruct(data, key)
            return data
        else:
            super().__getattr__(key)

    def __getitem__(self, key: int):
        if isinstance(key, int):
            return MatStruct(self.data[key], name=self.name)
        else:
            raise KeyError(f"invalid index: {key}. index must be an integer.")

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
        html_repr += self._generate_html_repr(self.data, is_field=True)
        html_repr += "</div>"
        return html_repr

    def _generate_html_repr(self, fields, level=0, access_code="", is_field=False):
        """Recursively generates HTML representation for fields."""
        html_repr = ""

        if isinstance(fields, (np.ndarray, np.void)) and (
            fields.dtype.names is not None
        ):
            for name in fields.dtype.names:
                value = fields[name]
                if isinstance(value, np.ndarray):
                    value = np.stack(value).squeeze()
                current_access_code = (
                    f"{access_code}.{name}" if is_field else f"{access_code}['{name}']"
                )
                html_repr += self._generate_field_html(
                    name, value, level, current_access_code
                )

        if isinstance(fields, dict):
            for key, value in fields.items():
                current_access_code = (
                    f"{access_code}.{key}" if is_field else f"{access_code}['{key}']"
                )
                html_repr += self._generate_field_html(
                    key, value, level, current_access_code
                )

        elif isinstance(fields, list) or (
            isinstance(fields, np.ndarray) and np.issubdtype(fields.dtype, np.str_)
        ):
            for index, item in enumerate(fields):
                access_code += f"[{index}]"
                html_repr += self._generate_field_html(index, item, level, access_code)
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
            if (value.dtype.names is not None) or np.issubdtype(value.dtype, np.str_):
                # If the value is a structured array, we generate HTML for its fields
                html_content = self._generate_html_repr(
                    value, level + 1, access_code, is_field=False
                )
            else:
                html_content = self._generate_array_html(value, level + 1)
        elif isinstance(value, (list, dict)):
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
            v = struct.data[k]
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
