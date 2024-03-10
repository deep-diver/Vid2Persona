import base64

def get_base64_content(file_path, decode=True):
    with open(file_path, 'rb') as f:
        data = f.read()

    return base64.b64decode(base64.b64encode(data)) if decode else base64.b64encode(data)
