"""Best-effort cleanup without losing the original operation failure."""


def cleanup_all(actions, primary=None):
    errors = []
    for name, action in actions:
        try:
            action()
        except Exception as error:
            errors.append((name, error))
    if not errors:
        return
    failure = primary if primary is not None else errors[0][1]
    for name, error in errors:
        if hasattr(failure, 'add_note'):
            failure.add_note(f'Cleanup {name}: {type(error).__name__}: {error}')
    if primary is None:
        raise failure
