def rename_model(model, new_name):
    """
    This allows model.__class__.__name__ to return 'new_name'.
    """
    dynamic_class = type(new_name, (model.__class__,), {})
    model.__class__ = dynamic_class
    return model