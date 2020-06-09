import tvm
from tvm import relay
import numpy as np

def get_cifar_relay():
    # data_shape = (1, 32, 32, 4)
    # conv0_kernel_shape = (5, 5, 32, 4)
    # conv1_kernel_shape = (5, 5, 32, 32)
    # conv2_kernel_shape = (5, 5, 64, 32)
    data_shape = (1, 3, 32, 32)             #changed the number of channels from 4 to 3
    conv0_kernel_shape = (32, 3, 5, 5)      #changed the number of channels from 4 to 3
    conv1_kernel_shape = (32, 32, 5, 5)
    conv2_kernel_shape = (64, 32, 5, 5)
    data_layout = 'NCHW'
    # kernel_layouts = ['HWOI', 'HWOI', 'HWOI']
    kernel_layouts = ['OIHW', 'OIHW', 'OIHW']
    bias_add_axis = 1

    mod = relay.fromtext(f"""
    v0.0.4
    def @main(%data: Tensor[{data_shape}, uint8],
        %mean_data: Tensor[{data_shape}, uint8],
        %conv0_weight: Tensor[{conv0_kernel_shape}, int8],
        %conv0_bias: Tensor[(32), int8],
        %conv1_weight: Tensor[{conv1_kernel_shape}, int8],
        %conv1_bias: Tensor[(32), int8],
        %conv2_weight: Tensor[{conv2_kernel_shape}, int8],
        %conv2_bias: Tensor[(64), int8],
        %dense0_weight: Tensor[(10, 1024), int8],
        %dense0_bias: Tensor[(10), int8]) {{
      %0 = cast(cast(%data, "int16") - cast(%mean_data, "int16"), "int8");
      %1 = nn.conv2d(
             %0,
             %conv0_weight,
             padding=[2, 2],
             channels=32,
             kernel_size=[5, 5],
             data_layout="{data_layout}",
             kernel_layout="{kernel_layouts[0]}",
             out_dtype="int32");
      %2 = nn.bias_add(%1, cast(%conv0_bias, "int32"), axis={bias_add_axis});
      %3 = right_shift(%2, 9);
      %4 = cast(%3, "int8");
      %5 = nn.max_pool2d(%4,
             pool_size=[3, 3],
             strides=[2, 2],
             layout="{data_layout}",
             ceil_mode=True);
      %6 = nn.relu(%5);
      %7 = nn.conv2d(
             %6,
             %conv1_weight,
             padding=[2, 2],
             channels=32,
             kernel_size=[5, 5],
             data_layout="{data_layout}",
             kernel_layout="{kernel_layouts[1]}",
             out_dtype="int32");
      %8 = nn.bias_add(%7, cast(%conv1_bias, "int32"), axis={bias_add_axis});
      %9 = right_shift(%8, 9);
      %10 = cast(%9, "int8");
      %11 = nn.relu(%10);
      %12 = nn.avg_pool2d(%11,
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %13 = nn.conv2d(%12,
              %conv2_weight,
              padding=[2, 2],
              channels=64,
              kernel_size=[5, 5],
              data_layout="{data_layout}",
              kernel_layout="{kernel_layouts[2]}",
              out_dtype="int32");
      %14 = nn.bias_add(%13, cast(%conv2_bias, "int32"), axis={bias_add_axis});
      %15 = right_shift(%14, 9);
      %16 = cast(%15, "int8");
      %17 = nn.relu(%16);
      %18 = nn.avg_pool2d(%17,
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %19 = nn.batch_flatten(%18);
      %20 = nn.dense(%19, %dense0_weight, units=10, out_dtype="int32");
      %21 = nn.bias_add(%20, left_shift(cast(%dense0_bias, "int32"), 3), axis=-1);
      %22 = right_shift(%21, 5);
      cast(%22, "int8")
    }}
    """)

    # generate random params
    params = {}
    for param in mod['main'].params[1:]:
        shape = list(map(lambda x: x.value, param.checked_type.shape))
        dtype = param.checked_type.dtype
        if 'bias' in param.name_hint:
            result = tvm.nd.array(np.random.randint(-3, 3, size=shape, dtype=dtype), tvm.cpu(0))
        elif 'weight' in param.name_hint:
            result = tvm.nd.array(np.random.randint(-30, 30, size=shape, dtype=dtype), tvm.cpu(0))
        elif 'mean' in param.name_hint:
            result = tvm.nd.array(np.random.randint(130, 140, size=shape, dtype=dtype), tvm.cpu(0))
        else:
            assert False
        params[param.name_hint] = result
    
    return mod, params