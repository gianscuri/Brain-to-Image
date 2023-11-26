import pickle
from huggingface_hub import hf_hub_download
import torch


from braindecode.models import EEGNetv4

lista = ['AlexMI', 'BNCI2014001', 'BNCI2014004', 'BNCI2015001', 'BNCI2015004', 'Cho2017', 'GrosseWentrup2009', 'Lee2019_MI', 'Ofner2017', 'PhysionetMI', 'Schirrmeister2017', 'Shin2017A', 'Shin2017B', 'Weibo2014', 'Zhou2016']


# # download the model from the hub:
# path_kwargs = hf_hub_download(
#     repo_id='PierreGtch/EEGNetv4',
#     filename='EEGNetv4_PhysionetMI/kwargs.pkl',
# )

path_params = hf_hub_download(
    repo_id='PierreGtch/EEGNetv4',
    filename=f'EEGNetv4_{lista[8]}/model-params.pkl',
)
# with open(path_kwargs, 'rb') as f:
#     kwargs = pickle.load(f)
# module_cls = kwargs['module_cls']
# module_kwargs = kwargs['module_kwargs']

# # load the model with pre-trained weights:
# torch_module = module_cls(**module_kwargs)

net = EEGNetv4(n_chans=3, n_times=384, n_outputs=7).eval() # ['C3', 'Cz', 'C4']
net.load_state_dict(torch.load(path_params, map_location='cuda'), strict=False)

                # channels=['C3', 'Cz', 'C4'],  # Same as the ones used to pre-train the embedding
                # events=['left_hand', 'right_hand', 'feet'],
                # n_classes=3,
                # fmin=0.5,
                # fmax=40,
                # tmin=0,
                # tmax=3,
                # resample=128,