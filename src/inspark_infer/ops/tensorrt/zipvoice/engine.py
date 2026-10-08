"""Original shared TensorRT binding and execution adapter."""
class Engine:

    def __init__(self, path, trt, torch, shared_workspace=False, input_shapes=None):
        import numpy as np
        self.trt, self.torch = (trt, torch)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Cannot deserialize {path}; rebuild for this host')
        strategy = trt.ExecutionContextAllocationStrategy.USER_MANAGED if shared_workspace else trt.ExecutionContextAllocationStrategy.STATIC
        self.context = self.engine.create_execution_context(strategy)
        if self.context is None:
            raise RuntimeError('Cannot allocate TensorRT execution context')
        dynamic = False
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            declared = tuple(self.engine.get_tensor_shape(name))
            if any((d < 0 for d in declared)):
                dynamic = True
                if not shared_workspace or input_shapes is None or name not in input_shapes:
                    raise RuntimeError('Dynamic engine requires explicit shapes and shared context workspace')
                if not self.context.set_input_shape(name, input_shapes[name]):
                    raise RuntimeError(f'Engine profile rejected the shape of {name}')
            elif input_shapes is not None and declared != tuple(input_shapes[name]):
                raise RuntimeError(f'Static engine does not match the workload for {name}')
        if dynamic:
            missing = self.context.infer_shapes()
            if missing:
                raise RuntimeError(f'Engine shape inference requires unbound inputs: {missing}')
        self.context_memory = int(self.context.update_device_memory_size_for_shapes()) if dynamic else int(self.engine.device_memory_size_v2)
        self.inputs, self.outputs, self.output_specs = ({}, {}, {})
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(f'External binding {name} requires unsupported non-LINEAR storage')
            shape = tuple(self.context.get_tensor_shape(name))
            if any((d < 0 for d in shape)):
                raise RuntimeError(f'Unresolved output shape: {name}')
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = (shape, dtype)
            else:
                self.output_specs[name] = (shape, dtype)
                if not shared_workspace:
                    self.outputs[name] = torch.empty(shape, dtype=dtype, device='cuda')

    def output_arena_size(self):
        import math
        offset = (self.context_memory + 255) // 256 * 256
        for shape, dtype in self.output_specs.values():
            offset += math.prod(shape) * self.torch.empty((), dtype=dtype).element_size()
            offset = (offset + 255) // 256 * 256
        return offset

    def bind_output_arena(self, arena):
        import math
        offset = (self.context_memory + 255) // 256 * 256
        for name, (shape, dtype) in self.output_specs.items():
            size = math.prod(shape) * self.torch.empty((), dtype=dtype).element_size()
            self.outputs[name] = arena.narrow(0, offset, size).view(dtype).reshape(shape)
            offset = (offset + size + 255) // 256 * 256
        if offset > arena.numel():
            raise RuntimeError('Output arena exceeds reserved storage')

    def __call__(self, values, stream):
        if set(values) != set(self.inputs):
            raise RuntimeError('Engine input names mismatch')
        for name, value in values.items():
            shape, dtype = self.inputs[name]
            if tuple(value.shape) != shape or value.dtype != dtype or (not value.is_cuda) or (not value.is_contiguous()):
                raise RuntimeError(f'Invalid binding {name}: shape/dtype/device/layout')
        for name, value in {**values, **self.outputs}.items():
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f'Cannot bind {name}')
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT enqueue failed')
        return self.outputs

