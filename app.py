import torch import numpy as np from PIL import Image import gradio as gr from Network import RAINet

class Args: pass

args = Args() args.S = 6 args.num_M = 32 args.num_Z = 32 args.T = 4

model = RAINet(args) state = torch.load("weights/RAINet.pth", map_location="cpu") model.load_state_dict(state) model.eval()

def pil_to_tensor(img): arr = np.array(img).astype(np.float32) / 255.0 arr = np.transpose(arr, (2, 0, 1)) return torch.tensor(arr).unsqueeze(0)

def tensor_to_pil(t): t = t.squeeze(0).detach().cpu().numpy() t = np.clip(t, 0, 1) t = np.transpose(t, (1, 2, 0)) return Image.fromarray((t * 255).astype(np.uint8))

def derain(img): x = pil_to_tensor(img) with torch.no_grad(): (output_tuple, _) = model(x) clean = output_tuple[0] return tensor_to_pil(clean)

demo = gr.Interface( fn=derain, inputs=gr.Image(type="pil"), outputs=gr.Image(type="pil"), title="RAINet Deraining Test" )

demo.launch()
