import abc
import math
import warnings
from enum import Enum

import numpy as np
import torch
from hyvideo.utils.ms_utils import no_grad


@no_grad
def find_event(interp_fn, sign0, t0, t1, event_fn, tol):
    # Num iterations for the secant method until tolerance is within target.
    nitrs = torch.ceil(torch.log((t1 - t0) / tol) / math.log(2.0))

    for _ in range(nitrs.long()):
        t_mid = (t1 + t0) / 2.0
        y_mid = interp_fn(t_mid)
        sign_mid = torch.sign(event_fn(t_mid, y_mid))
        same_as_sign0 = sign0 == sign_mid
        t0 = torch.where(same_as_sign0, t_mid, t0)
        t1 = torch.where(same_as_sign0, t1, t_mid)
    event_t = (t0 + t1) / 2.0

    return event_t, interp_fn(event_t)


class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(
        self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs
    ):
        self.atol = unused_kwargs.pop("atol")
        unused_kwargs.pop("rtol", None)
        unused_kwargs.pop("norm", None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {"callback_step"}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    solution[j] = self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j])
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1

        return solution

    def integrate_until_event(self, t0, event_fn):
        assert (
            self.step_size is not None
        ), "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = t1 - t0
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


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
    device = y0.device
    t_dtype = t0.dtype
    t0 = t0.to(t_dtype)

    if f0 is None:
        f0 = func(t0, y0)

    scale = atol + torch.abs(y0) * rtol

    d0 = norm(y0 / scale).abs()
    d1 = norm(f0 / scale).abs()

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=dtype, device=device)
    else:
        h0 = 0.01 * d0 / d1
    h0 = h0.abs()

    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)

    d2 = torch.abs(norm((f1 - f0) / scale) / h0)

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=dtype, device=device), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / float(order + 1))
    h1 = h1.abs()

    return torch.min(100 * h0, h1).to(t_dtype)


def _compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    error_tol = atol + rtol * torch.max(y0.abs(), y1.abs())
    return norm(error_estimate / error_tol).abs()


@torch.no_grad()
def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    if error_ratio == 0:
        return last_step * ifactor
    if error_ratio < 1:
        dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    error_ratio = error_ratio.type_as(last_step)
    exponent = torch.tensor(order, dtype=last_step.dtype, device=last_step.device).reciprocal()
    factor = torch.min(ifactor, torch.max(safety / error_ratio**exponent, dfactor))
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
    return torch.cat(tol)


def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)


class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return torch.cat([f_.reshape(-1) for f_ in f])


class _TupleInputOnlyFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleInputOnlyFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        return self.base_func(t, _flat_to_shape(y, (), self.shapes))


class _ReverseFunc(torch.nn.Module):
    def __init__(self, base_func, mul=1.0):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func
        self.mul = mul

    def forward(self, t, y):
        return self.mul * self.base_func(-t, y)


class Perturb(Enum):
    NONE = 0
    PREV = 1
    NEXT = 2


class _PerturbFunc(torch.nn.Module):
    def __init__(self, base_func):
        super(_PerturbFunc, self).__init__()
        self.base_func = base_func

    def forward(self, t, y, *, perturb=Perturb.NONE):
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
        raise NotImplementedError

    # Keep reference to original func as passed in
    original_func = func

    # Normalise to tensor (non-tupled) input
    shapes = None
    is_tuple = not isinstance(y0, torch.Tensor)
    if is_tuple:
        assert isinstance(y0, tuple), "y0 must be either a torch.Tensor or a tuple"
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol("rtol", rtol, shapes)
        atol = _tuple_tol("atol", atol, shapes)
        y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
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

    # Backward compatibility: Allow t and y0 to be on different devices
    if t.device != y0.device:
        warnings.warn("t is not on the same device as y0. Coercing to y0.device.")
        t = t.to(y0.device)
    # ~Backward compatibility

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
    def forward(ctx, x1, out):
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def _nextafter(x1, x2):
    with torch.no_grad():
        if hasattr(torch, "nextafter"):
            out = torch.nextafter(x1, x2)
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
    out = torch.tensor(np.nextafter(x1_np, x2_np)).to(x1)
    return out


def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, torch.Tensor), "{} must be a torch.Tensor".format(name)
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
        if isinstance(option_value, torch.Tensor):
            options[option_name] = -option_value
        # else: an error will be raised when the option is attempted to be used in Solver.__init__, but we defer raising
        # the error until then to keep things tidy.


SOLVERS = {}


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
        event_t = event_t.to(t)
        if t_is_reversed:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution
