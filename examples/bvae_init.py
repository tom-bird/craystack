import sys
import torch
import numpy as np
from autograd.builtins import tuple as ag_tuple
import craystack as cs
from torch_vae import BinaryVAE
from torch.distributions import Normal, Bernoulli
from torch_util import torch_fun_to_numpy_fun
from torchvision import datasets
import craystack.bb_ans as bb_ans
import time

rng = np.random.RandomState(0)

prior_precision = 10
bernoulli_precision = 16
q_precision = 14

zero_latents = False
do_map = True

num_images = 10000
num_pixels = num_images * 784
batch_size = 10  # single image
assert num_images % batch_size == 0

latent_dim = 40
latent_shape = (batch_size, latent_dim)
latent_size = np.prod(latent_shape)
obs_shape = (batch_size, 28 * 28)
obs_size = np.prod(obs_shape)

## Setup codecs
# VAE codec
model = BinaryVAE(hidden_dim=100, latent_dim=40)
model.load_state_dict(torch.load('vae_params'))

print('Loaded model with {} parameters'.format(sum([np.prod(p.shape) for p in model.parameters()])))


rec_net = torch_fun_to_numpy_fun(model.encode)
gen_net = torch_fun_to_numpy_fun(model.decode)

obs_codec = lambda p: cs.Bernoulli(p, bernoulli_precision)

# def vae_view(head):
#     return ag_tuple((np.reshape(head[:latent_size], latent_shape),
#                      np.reshape(head[latent_size:], obs_shape)))

# vae_append, vae_pop = cs.repeat(cs.substack(
#     bb_ans.VAE(gen_net, rec_net, obs_codec, prior_precision, q_precision),
#     vae_view), num_batches)

# init_append, init_pop = cs.substack(
#     bb_ans.VAE_init(gen_net, rec_net, obs_codec, prior_precision),
#     vae_view)

## Load mnist images
images = datasets.MNIST(sys.argv[1], train=False, download=True).data.numpy()
images = np.uint64(rng.random_sample(np.shape(images)) < images / 255.)
images = np.reshape(images, (num_images, -1))

init_images = images[:batch_size]
init_pixels = 784 * batch_size

# MAP training to find optimal latents
map_z = None
if do_map:
    map_z, _ = rec_net(init_images)  # start from the mean of the approximate posterior
    map_z = torch.from_numpy(map_z)
    map_z.requires_grad_()
    steps = 1000
    lr = 1e-3
    optimizer = torch.optim.Adam([map_z], lr=lr)
    init_images_torch = torch.from_numpy(init_images.astype(np.float32))


    def neg_log_joint(z):
        x_probs = model.decode(z)
        dist = Bernoulli(x_probs)
        l = torch.sum(dist.log_prob(init_images_torch), dim=1)
        p_z = torch.sum(Normal(0, 1).log_prob(z), dim=1)
        return -torch.mean(l + p_z) * np.log2(np.e) / 784.  # in bits per dim


    for step in range(steps):
        optimizer.zero_grad()
        loss = neg_log_joint(map_z)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Loss: {:.3f}'.format(loss))

    map_z = map_z.detach().numpy()


# set up codec
init_append, init_pop = bb_ans.VAE_init_1d(gen_net, rec_net, obs_codec, prior_precision, latent_shape,
                                           zero_latents, map_z)


## Encode
# Initialize message with some 'extra' bits
encode_t0 = time.time()
init_message = cs.base_message(1)  # just use a single element as the head
init_len = 32 * len(cs.flatten(init_message))

# Encode the mnist images
message, = init_append(init_message, init_images)

flat_message = cs.flatten(message)
encode_t = time.time() - encode_t0

print("All encoded in {:.2f}s.".format(encode_t))

message_len = 32 * len(flat_message)
print("Used {} bits.".format(message_len))
print("This is {:.4f} bits per pixel.".format(message_len / init_pixels))
print("Delta between init and msg len: {:.4f} bits per pixel.".format((message_len - init_len) / init_pixels))

## Decode
decode_t0 = time.time()
message = cs.unflatten(flat_message, (1,))

message, init_images_ = init_pop(message)
decode_t = time.time() - decode_t0

print('All decoded in {:.2f}s.'.format(decode_t))

np.testing.assert_equal(init_images, init_images_)
