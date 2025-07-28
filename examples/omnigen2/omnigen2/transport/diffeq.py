import warnings
from enum import Enum

import numpy as np

import mindspore as ms
from mindspore import mint, nn


def combine_event_functions(event_fn, t0, y0):
    """
    We ensure all event functions are initially positive,
    so then we can combine them by taking a min.
    """
    # with torch.no_grad(): TODO
    initial_signs = mint.sign(event_fn(t0, y0))

    def combined_event_fn(t, y):
        c = event_fn(t, y)
        return mint.min(c * initial_signs)

    return combined_event_fn


_all_callback_names = ["callback_step", "callback_accept_step", "callback_reject_step"]
_all_adjoint_callback_names = [name + "_adjoint" for name in _all_callback_names]
_null_callback = lambda *args, **kwargs: None


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn("{}: Unexpected arguments {}".format(solver.__class__.__name__, unused_kwargs))


def _linf_norm(tensor):
    return tensor.abs().max()


def _rms_norm(tensor):
    return tensor.abs().pow(2).mean().sqrt()


def _zero_norm(tensor):
    return 0.0


def _mixed_norm(tensor_tuple):
    if len(tensor_tuple) == 0:
        return 0.0
    return max([_rms_norm(tensor) for tensor in tensor_tuple])


def _select_initial_step(func, t0, y0, order, rtol, atol, norm, f0=None):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
    """

    dtype = y0.dtype
    t_dtype = t0.dtype
    t0 = t0.to(t_dtype)

    if f0 is None:
        f0 = func(t0, y0)

    scale = atol + mint.abs(y0) * rtol

    d0 = norm(y0 / scale).abs()
    d1 = norm(f0 / scale).abs()

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = ms.tensor(1e-6, dtype=dtype)
    else:
        h0 = 0.01 * d0 / d1
    h0 = h0.abs()

    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)

    d2 = mint.abs(norm((f1 - f0) / scale) / h0)

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = mint.max(ms.tensor(1e-6, dtype=dtype), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / float(order + 1))
    h1 = h1.abs()

    return mint.min(100 * h0, h1).to(t_dtype)


def _compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    error_tol = atol + rtol * mint.max(y0.abs(), y1.abs())
    return norm(error_estimate / error_tol).abs()


# @torch.no_grad()  TODO
def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    if error_ratio == 0:
        return last_step * ifactor
    if error_ratio < 1:
        dfactor = mint.ones((), dtype=last_step.dtype)
    error_ratio = error_ratio.type_as(last_step)
    exponent = ms.tensor(order, dtype=last_step.dtype).reciprocal()
    factor = mint.min(ifactor, mint.max(safety / error_ratio**exponent, dfactor))
    return last_step * factor


def _decreasing(t):
    return (t[1:] < t[:-1]).all()


def _assert_one_dimensional(name, t):
    assert t.ndimension() == 1, "{} must be one dimensional".format(name)


def _assert_increasing(name, t):
    assert (t[1:] > t[:-1]).all(), "{} must be strictly increasing or decreasing".format(name)


def _assert_floating(name, t):
    if not torch.is_floating_point(t):
        raise TypeError("`{}` must be a floating point Tensor but is a {}".format(name, t.type()))


def _tuple_tol(name, tol, shapes):
    try:
        iter(tol)
    except TypeError:
        return tol
    tol = tuple(tol)
    assert len(tol) == len(shapes), "If using tupled {} it must have the same length as the tuple y0".format(name)
    tol = [torch.as_tensor(tol_).expand(shape.numel()) for tol_, shape in zip(tol, shapes)]
    return mint.cat(tol)


def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)


class _TupleFunc(nn.Cell):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def construct(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return mint.cat([f_.reshape(-1) for f_ in f])


class _TupleInputOnlyFunc(nn.Cell):
    def __init__(self, base_func, shapes):
        super(_TupleInputOnlyFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def construct(self, t, y):
        return self.base_func(t, _flat_to_shape(y, (), self.shapes))


class _ReverseFunc(nn.Cell):
    def __init__(self, base_func, mul=1.0):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func
        self.mul = mul

    def construct(self, t, y):
        return self.mul * self.base_func(-t, y)


class Perturb(Enum):
    NONE = 0
    PREV = 1
    NEXT = 2


class _PerturbFunc(nn.Cell):
    def __init__(self, base_func):
        super(_PerturbFunc, self).__init__()
        self.base_func = base_func

    def construct(self, t, y, *, perturb=Perturb.NONE):
        assert isinstance(perturb, Perturb), "perturb argument must be of type Perturb enum"
        # This dtype change here might be buggy.
        # The exact time value should be determined inside the solver,
        # but this can slightly change it due to numerical differences during casting.
        if torch.is_complex(t):
            t = t.real
        t = t.to(y.abs().dtype)
        if perturb is Perturb.NEXT:
            # Replace with next smallest representable value.
            t = _nextafter(t, t + 1)
        elif perturb is Perturb.PREV:
            # Replace with prev largest representable value.
            t = _nextafter(t, t - 1)
        else:
            # Do nothing.
            pass
        return self.base_func(t, y)


def _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS):
    if event_fn is not None:
        if len(t) != 2:
            raise ValueError(f"We require len(t) == 2 when in event handling mode, but got len(t)={len(t)}.")

        # Combine event functions if the output is multivariate.
        event_fn = combine_event_functions(event_fn, t[0], y0)

    # Keep reference to original func as passed in
    original_func = func

    # Normalise to tensor (non-tupled) input
    shapes = None
    is_tuple = not isinstance(y0, ms.Tensor)
    if is_tuple:
        assert isinstance(y0, tuple), "y0 must be either a ms.Tensor or a tuple"
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol("rtol", rtol, shapes)
        atol = _tuple_tol("atol", atol, shapes)
        y0 = mint.cat([y0_.reshape(-1) for y0_ in y0])
        func = _TupleFunc(func, shapes)
        if event_fn is not None:
            event_fn = _TupleInputOnlyFunc(event_fn, shapes)

    # Normalise method and options
    if options is None:
        options = {}
    else:
        options = options.copy()
    if method is None:
        method = "dopri5"
    if method not in SOLVERS:
        raise ValueError(
            'Invalid method "{}". Must be one of {}'.format(method, '{"' + '", "'.join(SOLVERS.keys()) + '"}.')
        )

    if is_tuple:
        # We accept tupled input. This is an abstraction that is hidden from the rest of odeint (exception when
        # returning values), so here we need to maintain the abstraction by wrapping norm functions.

        if "norm" in options:
            # If the user passed a norm then get that...
            norm = options["norm"]
        else:
            # ...otherwise we default to a mixed Linf/L2 norm over tupled input.
            norm = _mixed_norm

        # In either case, norm(...) is assumed to take a tuple of tensors as input. (As that's what the state looks
        # like from the point of view of the user.)
        # So here we take the tensor that the machinery of odeint has given us, and turn it in the tuple that the
        # norm function is expecting.
        def _norm(tensor):
            y = _flat_to_shape(tensor, (), shapes)
            return norm(y)

        options["norm"] = _norm

    else:
        if "norm" in options:
            # No need to change the norm function.
            pass
        else:
            # Else just use the default norm.
            # Technically we don't need to set that here (RKAdaptiveStepsizeODESolver has it as a default), but it
            # makes it easier to reason about, in the adjoint norm logic, if we know that options['norm'] is
            # definitely set to something.
            options["norm"] = _rms_norm

    # Normalise time
    _check_timelike("t", t, True)
    t_is_reversed = False
    if len(t) > 1 and t[0] > t[1]:
        t_is_reversed = True

    if t_is_reversed:
        # Change the integration times to ascending order.
        # We do this by negating the time values and all associated arguments.
        t = -t

        # Ensure time values are un-negated when calling functions.
        func = _ReverseFunc(func, mul=-1.0)
        if event_fn is not None:
            event_fn = _ReverseFunc(event_fn)

        # For fixed step solvers.
        try:
            _grid_constructor = options["grid_constructor"]
        except KeyError:
            pass
        else:
            options["grid_constructor"] = lambda func, y0, t: -_grid_constructor(func, y0, -t)

        # For RK solvers.
        _flip_option(options, "step_t")
        _flip_option(options, "jump_t")

    # Can only do after having normalised time
    _assert_increasing("t", t)

    # Tol checking
    if torch.is_tensor(rtol):
        assert not rtol.requires_grad, "rtol cannot require gradient"
    if torch.is_tensor(atol):
        assert not atol.requires_grad, "atol cannot require gradient"

    # Add perturb argument to func.
    func = _PerturbFunc(func)

    # Add callbacks to wrapped_func
    callback_names = set()
    for callback_name in _all_callback_names:
        try:
            callback = getattr(original_func, callback_name)
        except AttributeError:
            setattr(func, callback_name, _null_callback)
        else:
            if callback is not _null_callback:
                callback_names.add(callback_name)
                # At the moment all callbacks have the arguments (t0, y0, dt).
                # These will need adjusting on a per-callback basis if that changes in the future.
                if is_tuple:

                    def callback(t0, y0, dt, _callback=callback):
                        y0 = _flat_to_shape(y0, (), shapes)
                        return _callback(t0, y0, dt)

                if t_is_reversed:

                    def callback(t0, y0, dt, _callback=callback):
                        return _callback(-t0, y0, dt)

            setattr(func, callback_name, callback)
    for callback_name in _all_adjoint_callback_names:
        try:
            callback = getattr(original_func, callback_name)
        except AttributeError:
            pass
        else:
            setattr(func, callback_name, callback)

    invalid_callbacks = callback_names - SOLVERS[method].valid_callbacks()
    if len(invalid_callbacks) > 0:
        warnings.warn("Solver '{}' does not support callbacks {}".format(method, invalid_callbacks))

    return shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed


class _StitchGradient(torch.autograd.Function):
    @staticmethod
    def construct(ctx, x1, out):
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def _nextafter(x1, x2):
    # with torch.no_grad():     TODO
    if hasattr(torch, "nextafter"):
        out = mint.nextafter(x1, x2)
    else:
        out = np_nextafter(x1, x2)
    return _StitchGradient.apply(x1, out)


def np_nextafter(x1, x2):
    warnings.warn(
        "torch.nextafter is only available in PyTorch 1.7 or newer."
        "Falling back to numpy.nextafter. Upgrade PyTorch to remove this warning."
    )
    x1_np = x1.detach().cpu().numpy()
    x2_np = x2.detach().cpu().numpy()
    out = ms.tensor(np.nextafter(x1_np, x2_np)).to(x1.dtype)
    return out


def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, ms.Tensor), "{} must be a ms.Tensor".format(name)
    _assert_floating(name, timelike)
    assert timelike.ndimension() == 1, "{} must be one dimensional".format(name)
    if not can_grad:
        assert not timelike.requires_grad, "{} cannot require gradient".format(name)
    diff = timelike[1:] > timelike[:-1]
    assert diff.all() or (~diff).all(), "{} must be strictly increasing or decreasing".format(name)


def _flip_option(options, option_name):
    try:
        option_value = options[option_name]
    except KeyError:
        pass
    else:
        if isinstance(option_value, ms.Tensor):
            options[option_name] = -option_value
        # else: an error will be raised when the option is attempted to be used in Solver.__init__, but we defer raising
        # the error until then to keep things tidy.


def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`, in either increasing or decreasing order. The first element of
            this sequence is taken to be the initial time point.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        event_fn: Function that maps the state `y` to a Tensor. The solve terminates when
            event_fn evaluates to zero. If this is not None, all but the first elements of
            `t` are ignored.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """

    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(
        func, y0, t, rtol, atol, method, options, event_fn, SOLVERS
    )

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)

    if event_fn is None:
        solution = solver.integrate(t)
    else:
        event_t, solution = solver.integrate_until_event(t[0], event_fn)
        event_t = event_t.to(t.dtype)
        if t_is_reversed:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution
