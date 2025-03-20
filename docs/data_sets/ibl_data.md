```python
import pynacollada as nac

from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
```

```python
print(one.search_terms())
```

```python
eeids, info = one.search(lab="witten", details=True)
```

```python
info[21]
```

```python

```
