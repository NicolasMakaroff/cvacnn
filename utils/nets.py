"""UNet model (http://arxiv.org/abs/1505.04597)."""

import types
from typing import Mapping, Optional, Sequence, Union, Any, Tuple

import haiku as hk

from functools import partial 
from jax import jit
import jax
from jax import lax
import jax.numpy as jnp
from matplotlib.style import use
import numpy as np
import copy

from utils.ccv import chan_vese, conv_phi

FloatStrOrBool = Union[str, float, bool]

class DeConv3x3(hk.Module):

    def __init__(self,
        channels: int,
        padding: str = 'SAME',
        use_batch_norm: bool = True,
        bn_config: Mapping[str, FloatStrOrBool] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        #if padding == 'SAME':
        #    padding = ((1, 2), (1, 2))
        self.use_batch_norm = use_batch_norm
        self.deconv = hk.Conv2DTranspose(output_channels=channels, stride=2, kernel_shape=(2,2), padding=padding)
        if use_batch_norm:
            self.bn_0 = hk.BatchNorm(name="deconv_batchnorm", **bn_config)

    def __call__(self, input, is_training):
        x = self.deconv(input)
        if self.use_batch_norm:
            x = self.bn_0(x, is_training)
        return x

class ConvRelu2(hk.Module):

    def __init__(self,
        channels: int,
        padding: str = 'SAME',
        kernel_shape: Tuple[int] = (3, 3),
        use_batch_norm: bool = True,
        bn_config: Mapping[str, FloatStrOrBool] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.use_batch_norm = use_batch_norm
        self.conv_0 = hk.Conv2D(output_channels=channels, kernel_shape=kernel_shape, padding=padding)
        if use_batch_norm:
            self.bn_0 = hk.BatchNorm(name="batchnorm0", **bn_config)
            self.bn_1 = hk.BatchNorm(name="batchnorm1", **bn_config)
        self.conv_1 = hk.Conv2D(output_channels=channels, kernel_shape=kernel_shape, padding=padding)

    def __call__(self, input: jnp.array, is_training: bool) -> jnp.array:
        x = self.conv_0(input)
        if self.use_batch_norm:
            x = self.bn_0(x, is_training)
        x = jax.nn.relu(x)
        x = self.conv_1(x)
        if self.use_batch_norm:
            x = self.bn_1(x, is_training)
        x = jax.nn.relu(x)
        return x

class DownsampleBlock(hk.Module):

    def __init__(self,
        channels: int,
        padding: str = 'SAME',
        use_batch_norm: bool = True,
        bn_config: Mapping[str, FloatStrOrBool] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.conv_relu = ConvRelu2(channels=channels, padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config)
        #self.max_pool = hk.MaxPool(window_shape=2, strides=2, padding=padding)

    def __call__(self, input: jnp.array, is_training: bool) -> jnp.array:
        residual = x = self.conv_relu(input, is_training=is_training)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='SAME',channel_axis=-1) #self.max_pool(x)
        return x, residual

class BottleneckBlock(hk.Module):

    def __init__(self,
        channels: int,
        padding: str = 'SAME',
        use_batch_norm: bool = True,
        bn_config: Mapping[str, FloatStrOrBool] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.conv_relu = ConvRelu2(channels=channels, padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config)
        self.deconv2d = DeConv3x3(channels=channels, padding=padding, use_batch_norm=use_batch_norm, name='deconv_bottleneck', bn_config=bn_config)


    def __call__(self, input: jnp.array, is_training: bool) -> jnp.array:
        x = self.conv_relu(input, is_training=is_training)
        x = self.deconv2d(x, is_training=is_training)
        return x

def central_crop(inputs, target_shape):
  """Returns a central crop in axis (1, 2).
  Args:
    inputs: nd-array; Inputs in shape of `[bs, height, width, channels]'.
    target_shape: tuple(int); Target shape after crop.
  Returns:
    Cropped image.
  """
  h, w = target_shape[1:3]
  assert h <= inputs.shape[1], f'{h} > {inputs.shape[1]}'
  assert w <= inputs.shape[2], f'{w} > {inputs.shape[2]}'
  h0 = (inputs.shape[1] - h) // 2
  w0 = (inputs.shape[2] - w) // 2
  return inputs[:, h0:(h0 + h), w0:(w0 + w)]

class BboxBlock(hk.Module):
    
    def __init__(self, 
        block_size: Tuple[int, ...],
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.fc_0 = hk.Linear(128)
        self.fc_1 = hk.Linear(64)
        self.fc_2 = hk.Linear(32)
        self.fc_bbox = hk.Linear(4)
        self.fc_intensity = hk.Linear(2)
    
    def __call__(self, input: jnp.array) -> jnp.array:
        out = jnp.mean(input, axis=(1,2))
        out = jax.nn.relu(out)
        out = jax.nn.relu(self.fc_0(out))
        out = jax.nn.relu(self.fc_1(out))
        out = jax.nn.relu(self.fc_2(out))
        bbox_coord = jax.nn.sigmoid(self.fc_bbox(out))
        intensity = jax.nn.sigmoid(self.fc_intensity(out))
        return bbox_coord, intensity

class EuclideanDistanceTransform(hk.Module):

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, input: jnp.array):
        return conv_phi(input)


class AttentionBlock(hk.Module):

    def __init__(self,
        channels: int,
        padding: str = 'SAME',
        use_batch_norm: bool = True,
        bn_config: Mapping[str, FloatStrOrBool] = None, 
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.conv_0 = hk.Conv2D(output_channels=channels, padding=padding, kernel_shape=(1, 1))
        self.conv_1 = hk.Conv2D(output_channels=channels, padding=padding, kernel_shape=(1, 1))
        self.conv_2 = hk.Conv2D(output_channels=1, kernel_shape=(1, 1), padding=padding)
        self.conv_3 = hk.Conv2D(output_channels=1, kernel_shape=(1, 1), padding=padding)
        self.conv_4 = hk.Conv2D(output_channels=1, kernel_shape=(1, 1), padding=padding)
        self.edt = EuclideanDistanceTransform()
        self.bn_0 = hk.BatchNorm(name="batchnorm0", **bn_config)
        self.bn_1 = hk.BatchNorm(name="batchnorm1", **bn_config)
        self.bn_2 = hk.BatchNorm(name="batchnorm2", **bn_config)
        self.bn_3 = hk.BatchNorm(name="batchnorm3", **bn_config)
        self.bn_4 = hk.BatchNorm(name="batchnorm4", **bn_config)

    def __call__(self, input: jnp.array, residual: jnp.array, img: jnp.array, is_training: bool) -> jnp.array:
        x1 = self.bn_0(self.conv_0(input), is_training=is_training)
        x2 = self.bn_1(self.conv_1(residual),is_training=is_training)
        x = jax.nn.relu(x1 + x2)
        x = self.bn_2(self.conv_2(x), is_training=is_training)
        x = jax.nn.sigmoid(x) 
        img = jax.nn.sigmoid(jax.image.resize(img, x.shape, 'linear'))
        input = jax.nn.sigmoid(self.bn_4(self.conv_4(img + self.bn_3(self.conv_3(x2), is_training=is_training)), is_training=is_training))
        edt = self.edt(x)
        segmentation = jnp.clip(chan_vese(input, edt, max_iter=30, mu=.1, lambda1=1.,lambda2=1., dt=0.1), a_min=-1, a_max=1)
        return segmentation * residual, edt, segmentation, input

class UpsampleBlock(hk.Module):

    def __init__(self,
        channels: int,
        padding: str = 'SAME',
        use_batch_norm: bool = True,
        bn_config: Mapping[str, FloatStrOrBool] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.conv_relu = ConvRelu2(channels=channels, padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config)
        self.attention = AttentionBlock(channels=channels // 2, padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config)
        self.deconv2d = DeConv3x3(channels=channels // 2, padding=padding, use_batch_norm=use_batch_norm, name='deconv3x3', bn_config=bn_config)

    def __call__(self, input: jnp.array, residual: jnp.array, img: jnp.array, is_training: bool) -> jnp.array:
        if residual is not None:
            x = jnp.concatenate([input, central_crop(residual, input.shape)], axis=-1)
        x1 = self.conv_relu(x, is_training=is_training)
        y, edt, segmentation, attention = self.attention(x1, residual, img, is_training=is_training)
        x = jnp.concatenate([x1, y], axis=-1)
        x = self.deconv2d(x, is_training=is_training)
        
        return x, edt, segmentation, attention


class OutputBlock(hk.Module):

    def __init__(self,
        channels: int,
        num_classes: int,
        padding: str = 'SAME',
        use_batch_norm: bool = True,
        bn_config: Mapping[str, FloatStrOrBool] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.use_batch_norm = use_batch_norm
        self.conv_relu = ConvRelu2(channels=channels, padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config)
        self.conv2d = hk.Conv2D(output_channels=num_classes, kernel_shape=1, name="conv1x1")
        if use_batch_norm:
            self.bn_0 = hk.BatchNorm(name="batchnorm0", **bn_config)
        

    def __call__(self, input: jnp.array, is_training: bool) -> jnp.array:
        x = self.conv_relu(input, is_training=is_training)
        x = self.conv2d(x)
        if self.use_batch_norm:
            x = self.bn_0(x, is_training)

        return x


class UNet(hk.Module):

    def __init__(self,
        num_classes: int,
        block_size: Tuple[int, ...] = (32, 64, 128, 256),  #(16, 32, 64, 128), #
        padding: str = 'SAME',
        use_batch_norm: bool = True,
        bn_config: Mapping[str, FloatStrOrBool] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        down_conv = DownsampleBlock(channels=block_size[0], padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config, name='down_0')
        downlayers = (down_conv,)
        for i, channels in enumerate(block_size[1:]):
            down_conv = DownsampleBlock(channels=channels, padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config, name=f'down_{i+1}')
            downlayers = downlayers + (down_conv,)
        self.down_layers = downlayers
        self.bottleneck = BottleneckBlock(channels=2*block_size[-1], padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config, name='bottleneck')
        *upscaling_channels, final_channels = block_size[::-1]
        up_conv = UpsampleBlock(channels=upscaling_channels[0], padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config, name='up_0')
        uplayers = (up_conv,)
        for i, channels in enumerate(upscaling_channels[1:]):
            up_conv = UpsampleBlock(channels=channels, padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config, name=f'up_{i+1}')
            uplayers = uplayers + (up_conv,)
        self.up_layers = uplayers
        self.out_block = OutputBlock(channels=final_channels, num_classes=num_classes, padding=padding, use_batch_norm=use_batch_norm, bn_config=bn_config, name='output')

    def __call__(self,
                x: jnp.array,
                is_training: bool,) -> jnp.array:
        input = x
        skip_connections = []
        for down in self.down_layers:
            x, residual = down(x, is_training=is_training)
            skip_connections.append(residual)

        x = self.bottleneck(x, is_training=is_training)
        for up in self.up_layers:
            residual = skip_connections.pop()
            x, edt, segmentation, attention = up(x, residual, input, is_training=is_training)

        x = self.out_block(x, is_training=is_training)
        #jax.debug.print('x min: {x}', x=jnp.min(x))
        #jax.debug.print('x max: {x}', x=jnp.max(x))
        x = jax.nn.sigmoid(x)
        #jax.debug.print('sig x min: {x}', x=jnp.min(x))
        #jax.debug.print('sig x max: {x}', x=jnp.max(x))
        return x, edt, segmentation, attention

