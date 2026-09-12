import json
import os
import pandas as pd
from services.Entities import Question, Option, QuestionAttributes


def _parse_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in ('true', '1', 'yes', 'y')


def _parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _determine_question_type(q_obj):
    """Detect question type from the question object dict.
    Returns a string key: 'mcq', 'true_false', 'ordering', 'matching', 'drag_drop', 'fill_blank'.
    """
    q_type = q_obj.get('Type', '')
    mapping = {
        '0': 'mcq',
        '1': 'ordering',
        '2': 'matching',
        '3': 'fill_blank',
        '5': 'drag_drop',
        '6': 'true_false',
    }
    return mapping.get(str(q_type), 'mcq')


def _map_matching_pairs(q_obj, options_raw):
    """Build a single Option for MatchingPairs questions (Type 2).
    
    The JSON has LeftItems and RightArrays; we combine them into one
    option content string representing the matched pair at each index.
    """
    left_items = q_obj.get('LeftItems') or []
    right_items = q_obj.get('RightItems') or []
    if not left_items or not right_items:
        return []
    # Create one option per pair, or a single combined option;
    # here we create one option per pair for backward compat,
    # but the Question __getattr__ can aggregate if needed.
    options = []
    for i, left in enumerate(left_items):
        # Pair left item with corresponding right item (if exists)
        right = right_items[i] if i < len(right_items) else ''
        combined = f"{left} - {right}"
        options.append(Option(
            option_id=f"match_{i}",
            content=combined,
            is_correct=True,
            display_order=i,
        ))
    return options


def _map_drag_drop(q_obj):
    """Build a single Option for DragAndDrop questions (Type 5).
    
    The JSON has Answers (all possibilities) and CorrectAnswers (the right ones).
    We create one Option with the correct answers list in order.
    """
    correct_answers = q_obj.get('CorrectAnswers') or []
    answers = q_obj.get('Answers') or []
    if not correct_answers:
        return []
    # Build content string of correct answers in order
    content = ', '.join(str(a) for a in correct_answers)
    # Optionally include wrong answers too, but per user request we just
    # store the correct list as the option content.
    return [Option(
        option_id="drag_drop_correct",
        content=content,
        is_correct=True,
        display_order=0,
    )]


def _map_ordering(q_obj):
    """Build Options for Ordering questions (Type 1).
    
    The JSON has Items and CorrectOrder; we create options with display_order
    set to position indices. Correctness is in the sequence, handled elsewhere.
    """
    items = q_obj.get('Items') or []
    if not items:
        return []
    options = []
    for i, item in enumerate(items):
        options.append(Option(
            option_id=f"order_{i}",
            content=str(item),
            is_correct=(i < len(q_obj.get('CorrectOrder', []))),
            display_order=i,
        ))
    return options


def _map_true_false(q_obj):
    """Build two Options for TrueOrFalse questions (Type 6).
    
    The JSON has CorrectAnswer boolean; we create two options, one True one False,
    with is_correct set appropriately.
    """
    correct = q_obj.get('CorrectAnswer', False)
    return [
        Option(option_id="tf_true", content="True", is_correct=bool(correct), display_order=0),
        Option(option_id="tf_false", content="False", is_correct=not bool(correct), display_order=1),
    ]


def _map_mcq(q_obj, options_raw):
    """Build Options for MCQ questions (Type 0) using the existing logic.
    Kept for backward compatibility with the G7 banking format.
    """
    options = []
    for opt_rec in options_raw:
        if not isinstance(opt_rec, dict):
            continue
        options.append(Option(
            option_id=str(opt_rec.get('en_option_ID', '')),
            content=str(opt_rec.get('en_Option_Content', '')),
            is_correct=bool(opt_rec.get('Is_Correct', False)),
            display_order=_parse_int(opt_rec.get('Display_Order', 0)),
        ))
    return options


def load_questions_from_json(path) -> list[Question]:
    """Parse a question batch JSON into Question entities.

    Supports both the G7 banking format and the question_bank.json format
    which uses a "Type" field (0=MCQ, 1=Ordering, 2=MatchingPairs,
    3=FillBlank, 5=DragAndDrop, 6=TrueOrFalse) along with type-specific
    fields.
    """
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return []
    records = data.get('questions', data) if isinstance(data, dict) else data
    questions = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        q_obj = rec.get('question') or rec  # some formats nest under 'question'
        q_type = _determine_question_type(q_obj)
        
        # Extract common fields that may appear in either format
        cm = rec.get('curriculum_mapping') or {}
        skill_cluster_ids = cm.get('Skill_Cluster_IDs') or []
        if isinstance(skill_cluster_ids, str):
            try:
                skill_cluster_ids = json.loads(skill_cluster_ids)
            except (json.JSONDecodeError, TypeError):
                skill_cluster_ids = []
        unit_id = cm.get('Unit_ID')
        lesson_id = cm.get('Lesson_ID')
        skill_ids = [str(s) for s in skill_cluster_ids] if skill_cluster_ids else []
        skill_cluster_id = skill_ids[0] if skill_ids else None

        options = []
        # --- Branch by question type ---
        if q_type == 'matching':
            options = _map_matching_pairs(q_obj, [])
        elif q_type == 'drag_drop':
            options = _map_drag_drop(q_obj)
        elif q_type == 'ordering':
            options = _map_ordering(q_obj)
        elif q_type == 'true_false':
            options = _map_true_false(q_obj)
        else:
            # MCQ and any other type: use the existing options_raw logic
            options_raw = rec.get('options', [])
            if isinstance(options_raw, list):
                for opt_rec in options_raw:
                    if not isinstance(opt_rec, dict):
                        continue
                    options.append(Option(
                        option_id=str(opt_rec.get('en_option_ID', '')),
                        content=str(opt_rec.get('en_Option_Content', '')),
                        is_correct=bool(opt_rec.get('Is_Correct', False)),
                        display_order=_parse_int(opt_rec.get('Display_Order', 0)),
                    ))
        
        # Extract question diagnostics (G7 style) - try both formats
        q_diag = rec.get('question_diagnostics') or rec.get('question_diagnostics') or {}
        # Also try top-level if not nested
        if not q_diag and isinstance(rec, dict):
            for key in ('question_diagnostics', 'diagnostics'):
                if key in rec and isinstance(rec[key], dict):
                    q_diag = rec[key]
                    break
        
        attrs = QuestionAttributes(
            difficulty_level=_parse_float(q_diag.get('Difficulty_Level')),
            bloom_taxonomy_level=q_diag.get('Bloom_Taxonomy_Level'),
            cognitive_load_index=_parse_int(q_diag.get('Cognitive_Load_Index')),
            variables_count=_parse_int(q_diag.get('Variables_Count')),
            logical_steps=_parse_int(q_diag.get('Logical_Steps')),
            time_allowed=_parse_float(q_diag.get('Theoretical_Time')),
            theoretical_solving_time=_parse_float(q_diag.get('Theoretical_Time')),
            trick_question=_parse_bool(q_diag.get('Trick_Question')),
            visual_dependency=_parse_bool(q_diag.get('Visual_Dependency')),
            real_world_context=_parse_bool(q_diag.get('Real_World_Context')),
            multi_concept_flag=_parse_bool(q_diag.get('Multi_Concept_Flag')),
            prerequisite_concept_ids=[str(s) for s in (q_diag.get('Prerequisite_Concept_IDs') or [])],
        )
        
        questions.append(Question(
            id=str(q_obj.get('en_question_ID', q_obj.get('id', ''))),
            question_type=q_type,
            question_text=q_obj.get('Prompt') or q_obj.get('en_Qs_text', ''),
            grade=None,
            subject=None,
            unit_id=unit_id,
            lesson_id=lesson_id,
            skill_cluster_id=skill_cluster_id,
            skill_ids=skill_ids,
            options=options,
            attrs=attrs,
        ))
    return questions


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