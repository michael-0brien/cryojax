from collections.abc import Callable

import jax
import tqdm.auto
import tqdm.notebook
import tqdm.std
from jax.debug import callback


def fori_loop_tqdm_decorator(
    n_iterations: int,
    print_every: int | None = None,
    **kwargs,
) -> Callable:
    """Add a tqdm progress bar to `body_fun` used in `jax.lax.fori_loop`.
    This function is closely based on the implementation in
    [`jax_tqdm`](https://github.com/jeremiecoullon/jax-tqdm).
    """

    _update_progress_bar, close_tqdm = _build_tqdm(n_iterations, print_every, **kwargs)

    def _fori_loop_tqdm_decorator(func):
        def wrapper_progress_bar(i, val):
            _update_progress_bar(i)
            result = func(i, val)
            return close_tqdm(result, i)

        return wrapper_progress_bar

    return _fori_loop_tqdm_decorator


def _build_tqdm(
    n: int,
    print_rate: int | None = None,
    tqdm_type: str = "auto",
    **kwargs,
) -> tuple[Callable, Callable]:
    if tqdm_type not in ("auto", "std", "notebook"):
        raise ValueError(
            'tqdm_type should be one of "auto", "std", or "notebook" '
            f'but got "{tqdm_type}"'
        )
    pbar = getattr(tqdm, tqdm_type).tqdm

    desc = kwargs.pop("desc", f"Running for {n:,} iterations")
    message = kwargs.pop("message", desc)
    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    tqdm_bars = {}

    if print_rate is None:
        if n > 20:
            print_rate = int(n / 20)
        else:
            print_rate = 1
    else:
        if print_rate < 1:
            raise ValueError(f"Print rate should be > 0 got {print_rate}")
        elif print_rate > n:
            raise ValueError(
                "Print rate should be less than the "
                f"number of steps {n}, got {print_rate}"
            )

    remainder = n % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = pbar(range(n), **kwargs)
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(int(arg))

    def _update_progress_bar(iter_num):
        "Updates tqdm from a JAX scan or loop"
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: callback(_define_tqdm, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != n - remainder),
            lambda _: callback(_update_tqdm, print_rate, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == n - remainder,
            lambda _: callback(_update_tqdm, remainder, None, ordered=True),
            lambda _: None,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        _ = jax.lax.cond(
            iter_num == n - 1,
            lambda _: callback(_close_tqdm, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )
        return result

    return _update_progress_bar, close_tqdm
