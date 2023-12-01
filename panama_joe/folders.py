import os

_OUTPUT_DIR = 'output'
_DEMOS_DIR = os.path.join(_OUTPUT_DIR, 'demos')
_MODELS_DIR = os.path.join(_OUTPUT_DIR, 'models')
_VIDEOS_DIR = os.path.join(_OUTPUT_DIR, 'videos')


def demo_dir(model_name):
    return _child_dir(_DEMOS_DIR, model_name)


def model_dir(model_name):
    return _child_dir(_MODELS_DIR, model_name)


def video_dir(model_name):
    return _child_dir(_VIDEOS_DIR, model_name)


def _child_dir(parent_dir, model_name):
    child_dir = os.path.join(parent_dir, model_name)
    os.makedirs(child_dir, exist_ok=True)
    return child_dir
