
def load_system_info(path : str = 'ProgAgents\info.txt') -> str:
    with open(path, encoding='utf-8') as f:
        content = f.read()
    return content

load_system_info()