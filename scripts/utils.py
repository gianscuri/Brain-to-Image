import os
import torch

def save_model(model, metric, name):
    # Save model
    save_dir = os.path.join('../trained_models', name)
    save_file = os.path.join(save_dir, f'model_{metric:04.1f}.pt')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, save_file)
    print('Model saved!')

def save_checkpoint(model, file_name, new_metric, best_metric):
    if new_metric > best_metric:
        best_metric = new_metric
        save_dir = os.path.join('../trained_models', file_name)
        save_file = os.path.join(save_dir, 'checkpoint.pt')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_file)
    return best_metric

def rename_saved_model(file_name, metric):
    # Rename model file with accuracy
    old_file = os.path.join('../trained_models', file_name, 'checkpoint.pt')
    new_file = os.path.join('../trained_models', file_name, f'model_{metric:04.1f}.pt')
    os.rename(old_file, new_file)


def best_epoch(model, metric, best_metric):
    if metric > best_metric:
        best_metric = metric
        best_model = model

        return best_metric, best_model
    
    
# def load_best_model(model, file_name):
#     # Load model
#     load_dir = os.path.join('../trained_models', file_name)
#     load_file = os.path.join(load_dir, 'checkpoint.pt')
#     model.load_state_dict(torch.load(load_file))
#     return model

# def load_model(model, file_name):
#     # Load model
#     load_dir = os.path.join('../trained_models', file_name)
#     load_file = os.path.join(load_dir, 'checkpoint.pt')
#     model.load_state_dict(torch.load(load_file))
#     return model