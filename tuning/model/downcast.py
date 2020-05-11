import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.expr import Call, Var, Constant, TupleGetItem
from tvm.relay import Function
from tvm.relay import transform as _transform
from tvm.ir import IRModule
from tvm.relay import cast

def downcast_int8(module):
    # pylint: disable=line-too-long
    """Downcast to int8 mutator
    Parameters
    ---------
    graph: Function
        The original graph.
    Retruns
    -------
    The graph after dowmcasting to half-precision floating-point.
    """
    # get_valid_counts and non_max_suppression does not support fp16 so we create a filter list for them
    # filter_list = ['nn.conv2d', 'nn.batch_norm', 'nn.dense']
    filter_list = []
    class DowncastMutator(ExprMutator):
        """Downcast to int8 mutator"""
        def visit_call(self, call):
            # dtype = 'int8' if call.op.name in filter_list else 'float32'
            dtype = 'int8'
            new_fn = self.visit(call.op)
            # Collec the original dtypes
            type_list = []
            if call.op.name in filter_list:
                # For nms
                for arg in call.args:
                    if isinstance(arg, TupleGetItem) and isinstance(arg.tuple_value, Call):
                        tuple_types = arg.tuple_value.checked_type.fields
                        type_list.append(tuple_types[arg.index].dtype)
                        import pdb; pdb.set_trace()
                if call.op.name == 'vision.get_valid_counts':
                    tuple_types = call.checked_type.fields
                    for cur_type in tuple_types:
                        type_list.append(cur_type.dtype)
            args = [self.visit(arg) for arg in call.args]
            new_args = list()
            arg_idx = 0
            for arg in args:
                if isinstance(arg, (Var, Constant)):
                    new_args.append(cast(arg, dtype=dtype))
                else:
                    if call.op.name in filter_list:
                        if isinstance(arg, TupleGetItem) and type_list[arg_idx] == 'float32':
                            new_args.append(arg)
                            import pdb; pdb.set_trace()
                        else:
                            new_args.append(cast(arg, dtype=dtype))
                    else:
                        new_args.append(arg)
                arg_idx += 1

            if call.op.name in filter_list and call.op.name != 'nn.batch_norm':
                return cast(Call(new_fn, new_args, call.attrs), dtype='int8')

            return Call(new_fn, new_args, call.attrs)

    # class UpcastMutator(ExprMutator):
    #     """upcast output back to fp32 mutator"""
    #     def visit_call(self, call):
    #         return cast(call, dtype='float32')
    
    def infer_type(mod):
        """A method to infer the type of an intermediate node in the relay graph"""
        mod = _transform.InferType()(mod)
        entry = mod["main"]
        if isinstance(entry, Function):
            return entry
        else:
            raise TypeError('Should be function!')

    func = infer_type(module)
    downcast_pass = DowncastMutator()
    func = downcast_pass.visit(func)

    # upcast_pass = UpcastMutator()
    # func = upcast_pass.visit(func)
    # func = infer_type(func)
    return func