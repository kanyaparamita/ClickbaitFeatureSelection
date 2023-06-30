import pandas as pd

def load_txt_data(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as file:
        content = file.read()

    sentences = content.split('\n\n')
    data = {'headline': sentences}
    return pd.DataFrame(data)