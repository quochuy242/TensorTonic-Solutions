TYPE_MAP = {
    "str": (str, ),
    "int": (int, ),
    "float": (float, int),
}

def map_schema(schema):
    schema_map = {
        s['column']: s
        for s in schema
    }
    return schema_map

def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    # Write code here
    schema = map_schema(schema)

    errors = []

    for i, record in enumerate(records):
        record_ok = True 
        record_errors = []
        
        for column, rules in schema.items():
            # Missing column
            exist = column in record
            if not exist:
                record_ok = False
                record_errors.append(f"{column}: missing")

            value = record.get(column, None)
            is_null = value is None and exist
            
            # Null-check 
            if is_null and not rules['nullable']:
                record_ok = False
                record_errors.append(f"{column}: null")

            # Type check
            type_val = type(value).__name__
            if not isinstance(value, TYPE_MAP[rules['type']]) and not type_val == "NoneType":
                record_ok = False
                record_errors.append(f"{column}: expected {rules['type']}, got {type_val}")

            # Range check
            if isinstance(value, TYPE_MAP['float']):
                if not (rules['min'] <= value <= rules['max']):
                    record_ok = False
                    record_errors.append(f"{column}: out of range")
                    
        errors.append([i, record_ok, record_errors])

    return errors