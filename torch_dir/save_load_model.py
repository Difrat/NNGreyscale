import torch


def save_model(model, path_to_file:str) -> None:
    st = model.state_dict()
    torch.save(st, f'{path_to_file}')
    return None


def get_state_dict(path_to_file:str):
    state_dict = torch.load(path_to_file, weights_only=True)
    return state_dict