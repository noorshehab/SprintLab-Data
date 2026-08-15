import json
import os
import pandas as pd


def load_records(path, fmt=None):
    """Load a data file into a list of dicts (csv) or a dict keyed by id (json).

    JSON files are expected to be either {id: {fields}} (e.g. questions.json) or
    a plain list of dicts.
    """
    if fmt is None:
        ext = str(path).lower()
        fmt = 'json' if ext.endswith('.json') else 'csv'
    if fmt == 'csv':
        df = pd.read_csv(path)
        return df.to_dict('records')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_ids(records, id_field, id_key=None, id_type=int):
    """Extract unique id values from a loaded structure.

    records: list of dicts (csv rows / json list) OR dict keyed by id (json map).
    id_key: when records is a json map, key name inside each value to collect ids from
            (e.g. the 'questions' list of a responses row -> student ids).
            If None, the map keys themselves are the ids.
    """
    ids = set()
    if isinstance(records, dict):
        for k, v in records.items():
            if id_key is None:
                ids.add(k)
            else:
                for item in _nested_values(v.get(id_key)):
                    ids.add(item)
    else:
        for row in records:
            if id_field in row:
                ids.add(row[id_field])
    result = []
    for v in ids:
        try:
            result.append(id_type(v))
        except (TypeError, ValueError):
            result.append(v)
    return result


def _nested_values(value):
    """Yield scalar items from a scalar, list, or list-of-lists value."""
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_values(item)
    else:
        yield value


def map_row(row, field_map, defaults=None):
    """Map raw record fields to the canonical target names.

    field_map: {target_name: source_name}
    defaults:  {target_name: default_value} for missing sources
    """
    defaults = defaults or {}
    out = {}
    for target, source in field_map.items():
        if source in row and pd.notna(row[source]):
            out[target] = row[source]
        elif target in defaults:
            out[target] = defaults[target]
    return out


def load_skill_texts(path, id_field='kc_id', text_field='kc_route', fmt=None):
    """Load skill text keyed by skill id for embedding.

    Returns {skill_id: text} from a csv (e.g. kc_metadata.csv with the kc_route
    column) or json map.
    """
    records = load_records(path, fmt=fmt)
    out = {}
    if isinstance(records, dict):
        for k, v in records.items():
            text = v.get(text_field) if isinstance(v, dict) else v
            out[k] = '' if text is None else str(text)
    else:
        for row in records:
            if id_field not in row:
                continue
            text = row.get(text_field)
            out[row[id_field]] = '' if text is None else str(text)
    return out