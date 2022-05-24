from craystack.codecs import substack, Uniform, \
    std_gaussian_centres, DiagGaussian_StdBins, Codec, std_gaussian_buckets
import numpy as np
from scipy.stats import norm
import craystack as cs


def BBANS(prior, likelihood, posterior):
    """
    This codec is for data modelled with a latent variable model as described
    in the paper 'Practical Lossless Compression with Latent Variable Models'
    currently under review for ICLR '19.

       latent        observed
      variable         data

        ( z ) ------> ( x )

    This assumes data x is modelled via a model which includes a latent
    variable. The model has a prior p(z), likelihood p(x | z) and (possibly
    approximate) posterior q(z | x). See the paper for more details.
    """
    prior_push, prior_pop = prior

    def push(message, data):
        _, posterior_pop = posterior(data)
        message, latent = posterior_pop(message)
        likelihood_push, _ = likelihood(latent)
        message, = likelihood_push(message, data)
        message, = prior_push(message, latent)
        return message,

    def pop(message):
        message, latent = prior_pop(message)
        likelihood_pop = likelihood(latent).pop
        message, data = likelihood_pop(message)
        posterior_push = posterior(data).push
        message, = posterior_push(message, latent)
        return message, data
    return Codec(push, pop)


def VAE(gen_net, rec_net, obs_codec, prior_prec, latent_prec):
    """
    This codec uses the BB-ANS algorithm to code data which is distributed
    according to a variational auto-encoder (VAE) model. It is assumed that the
    VAE uses an isotropic Gaussian prior and diagonal Gaussian for its
    posterior.
    """
    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    prior = substack(Uniform(prior_prec), z_view)

    def likelihood(latent_idxs):
        z = std_gaussian_centres(prior_prec)[latent_idxs]
        return substack(obs_codec(gen_net(z)), x_view)

    def posterior(data):
        post_mean, post_stdd = rec_net(data)
        return substack(DiagGaussian_StdBins(
            post_mean, post_stdd, latent_prec, prior_prec), z_view)
    return BBANS(prior, likelihood, posterior)


def BBANS_init(prior, likelihood, posterior_sampler):
    """
    This codec is for data modelled with a latent variable model as described
    in the paper 'Practical Lossless Compression with Latent Variable Models'
    currently under review for ICLR '19.

       latent        observed
      variable         data

        ( z ) ------> ( x )

    This assumes data x is modelled via a model which includes a latent
    variable. The model has a prior p(z), likelihood p(x | z) and (possibly
    approximate) posterior q(z | x). See the paper for more details.
    """
    prior_push, prior_pop = prior

    def push(message, data):
        latent = posterior_sampler(data)
        likelihood_push, _ = likelihood(latent)
        message, = likelihood_push(message, data)
        message, = prior_push(message, latent)
        return message,

    def pop(message):
        message, latent = prior_pop(message)
        likelihood_pop = likelihood(latent).pop
        message, data = likelihood_pop(message)
        return message, data
    return Codec(push, pop)


def VAE_init(gen_net, rec_net, obs_codec, prior_prec):
    """
    This codec uses the BB-ANS algorithm to code data which is distributed
    according to a variational auto-encoder (VAE) model. It is assumed that the
    VAE uses an isotropic Gaussian prior and diagonal Gaussian for its
    posterior.
    """
    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    rng = np.random.RandomState(1)

    prior = substack(Uniform(prior_prec), z_view)

    def likelihood(latent_idxs):
        z = std_gaussian_centres(prior_prec)[latent_idxs]
        return substack(obs_codec(gen_net(z)), x_view)

    def posterior_sampler(data):
        post_mean, post_stdd = rec_net(data)
        eps = rng.randn(*post_mean.shape)
        z = post_mean + eps * post_stdd  # cts, need to discretise according to prior
        # work out cts proxy
        bits = - norm.logpdf(z).sum() / np.log(2.)
        print(bits)

        idxs = np.uint64(np.digitize(z, std_gaussian_buckets(prior_prec)) - 1)
        return idxs

    return BBANS_init(prior, likelihood, posterior_sampler)


def BBANS_init_1d(prior, likelihood, posterior_sampler, latent_shape, zero_latents):
    """
    This codec is for data modelled with a latent variable model as described
    in the paper 'Practical Lossless Compression with Latent Variable Models'
    currently under review for ICLR '19.

       latent        observed
      variable         data

        ( z ) ------> ( x )

    This assumes data x is modelled via a model which includes a latent
    variable. The model has a prior p(z), likelihood p(x | z) and (possibly
    approximate) posterior q(z | x). See the paper for more details.
    """
    prior_push, prior_pop = prior

    def push(message, data):
        if zero_latents:
            latent = np.zeros(latent_shape, dtype=np.uint64)
        else:
            latent = posterior_sampler(data)
        likelihood_push, _ = likelihood(latent)
        flat_data = np.ravel(data)  # flatten
        def count(msg):
            print(32 * len(cs.flatten(msg)))
        count(message)
        message, = likelihood_push(message, flat_data)
        count(message)
        if not zero_latents:
            message = cs.reshape_head(message, latent_shape)
            message, = prior_push(message, latent)
            message = cs.reshape_head(message, (1,))
            count(message)
        return message,

    def pop(message):
        if zero_latents:
            latent = np.zeros(latent_shape, dtype=np.uint64)
        else:
            message = cs.reshape_head(message, latent_shape)
            message, latent = prior_pop(message)
            message = cs.reshape_head(message, (1,))
        likelihood_pop = likelihood(latent).pop
        message, data = likelihood_pop(message)
        data = np.reshape(data, (latent_shape[0], -1))
        return message, data
    return Codec(push, pop)

def VAE_init_1d(gen_net, rec_net, obs_codec, prior_prec, latent_shape, zero_latents=False, map_z=None):
    """
    This codec uses the BB-ANS algorithm to code data which is distributed
    according to a variational auto-encoder (VAE) model. It is assumed that the
    VAE uses an isotropic Gaussian prior and diagonal Gaussian for its
    posterior.
    """
    rng = np.random.RandomState(1)

    prior = Uniform(prior_prec)

    def likelihood(latent_idxs):
        z = std_gaussian_centres(prior_prec)[latent_idxs]
        p = np.ravel(gen_net(z))  # flatten
        flat_obs_codec = cs.from_iterable([obs_codec(pi) for pi in p])
        return flat_obs_codec

    def posterior_sampler(data):
        if map_z is not None:
            z = map_z
        else:
            post_mean, post_stdd = rec_net(data)
            eps = rng.randn(*post_mean.shape)
            z = post_mean + eps * post_stdd  # cts, need to discretise according to prior
        # work out cts proxy
        bits = - norm.logpdf(z).sum() / np.log(2.)
        bbp = bits / np.prod(data.shape)
        print('Cts proxy bits per pixel to code z: {:.3f}'.format(bbp))

        idxs = np.uint64(np.digitize(z, std_gaussian_buckets(prior_prec)) - 1)
        return idxs

    return BBANS_init_1d(prior, likelihood, posterior_sampler, latent_shape, zero_latents)
