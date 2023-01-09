# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

import numpy as np

def l2_normalize(x, epsilon=1e-12, axis=None):
    return x / np.sqrt(np.maximum(np.sum(x ** 2, axis=axis), epsilon))

# @gin.configurable(blacklist=["inputs"])
def spectral_norm(inputs, epsilon=1e-12, singular_value="right", power_iteration_rounds=5):
    """Performs Spectral Normalization on a weight tensor.

    Details of why this is helpful for GAN's can be found in "Spectral
    Normalization for Generative Adversarial Networks", Miyato T. et al., 2018.
    [https://arxiv.org/abs/1802.05957].

    Args:
      inputs: The weight tensor to normalize.
      epsilon: Epsilon for L2 normalization.
      singular_value: Which first singular value to store (left or right). Use
        "auto" to automatically choose the one that has fewer dimensions.

    Returns:
      The normalized weight tensor.
    """
    if len(inputs.shape) <= 0:
        # logging.info("[ops] spectral norm of a float is itself; returning as-is. name=%s %s", inputs.name, repr(inputs))
        return inputs
    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). Our Conv2D kernel shape is (KH, KW, C_in, C_out)
    # so it should be reshaped to (KH * KW * C_in, C_out), and similarly for other
    # layers that put output channels as last dimension. This implies that w
    # here is equivalent to w.T in the paper.
    w = inputs.reshape((-1, inputs.shape[-1]))

    # Choose whether to persist the first left or first right singular vector.
    # As the underlying matrix is PSD, this should be equivalent, but in practice
    # the shape of the persisted vector is different. Here one can choose whether
    # to maintain the left or right one, or pick the one which has the smaller
    # dimension. We use the same variable for the singular vector if we switch
    # from normal weights to EMA weights.
    if singular_value == "auto":
        singular_value = "left" if w.shape[0] <= w.shape[1] else "right"
    u_shape = (w.shape[0], 1) if singular_value == "left" else (1, w.shape[-1])
    u = np.random.normal(size=u_shape)

    # Use power iteration method to approximate the spectral norm.
    # The authors suggest that one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance.
    for _ in range(power_iteration_rounds):
        if singular_value == "left":
            # `v` approximates the first right singular vector of matrix `w`.
            v = l2_normalize(
                np.matmul(np.transpose(w), u), axis=None, epsilon=epsilon)
            u = l2_normalize(np.matmul(w, v), axis=None, epsilon=epsilon)
        else:
            v = l2_normalize(np.matmul(u, np.transpose(w)), epsilon=epsilon)
            u = l2_normalize(np.matmul(v, w), epsilon=epsilon)

    # The authors of SN-GAN chose to stop gradient propagating through u and v
    # and we maintain that option.
    # u = tf.stop_gradient(u)
    # v = tf.stop_gradient(v)

    if singular_value == "left":
        norm_value = np.matmul(np.matmul(np.transpose(u), w), v)
    else:
        norm_value = np.matmul(np.matmul(v, w), np.transpose(u))
    # norm_value.shape.assert_is_fully_defined()
    # norm_value.shape.assert_is_compatible_with([1, 1])
    return norm_value[0][0]
