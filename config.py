
class GlobalConfig:
    """ Main architecture parameters """
 
    seq_len = 5 # input timesteps
    root = './Dataset'

    n_views = 2 # no. of camera views
    n_gps = 2 # no. of gps views
    
    input_resolution = 256 # valid for GPT transfuser
    input_resolution = (25,20) # valid for SWIN transfuser

    scale = 1 # image pre-processing
    crop = 224 # image pre-processing

    lr = 1e-3 # learning rate

    # Conv Encoder
    vert_anchors = 7
    horz_anchors = 7
    anchors = vert_anchors * horz_anchors

	# GPT Encoder
    n_embd = 192
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

