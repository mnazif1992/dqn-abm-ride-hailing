"""
ماژول monkey-patching استراتژی‌های تخصیص dispatcher (مرحله ۵).

این ماژول یک context manager فراهم می‌کند که DispatcherAgent.dispatch_step
را با یکی از سه استراتژی baseline (random, greedy, hungarian) جایگزین می‌کند،
بدون هیچ تغییری در src/abm/.

اصول طراحی:
    - بدون تغییر در src/abm/ — همه‌چیز monkey-patching است
    - context manager: تضمین بازگشت به حالت اولیه پس از خروج (try/finally)
    - سازگار با لایه‌های بیرونی patch (مثل no_driver_threshold در abm_param_overrides):
      اگر استراتژی *قبل* از abm_param_overrides اعمال شود، آن wrapper
      استراتژی ما را به‌عنوان "original" می‌گیرد و دور آن می‌پیچد.

ترتیب کاربرد توصیه‌شده:
    with dispatcher_strategy("hungarian"):
        result = run_multi_seed(
            base_config=config, targets=targets, zones_data=zones,
            overrides={"no_driver_threshold_steps": 2, ...},
            seeds=[42, 123, ...],
        )
    # خارج از بلوک: dispatch_step اصلی Greedy ABM بازگشته
"""
from __future__ import annotations

import contextlib
import logging
from typing import Iterator

logger = logging.getLogger(__name__)

VALID_STRATEGIES = ("random", "greedy", "hungarian", "hungarian_resolve")


@contextlib.contextmanager
def dispatcher_strategy(strategy_name: str) -> Iterator[None]:
    """
    Context manager برای جایگزینی DispatcherAgent.dispatch_step با یک baseline.

    ورودی:
        strategy_name: یکی از "random" | "greedy" | "hungarian" | "hungarian_resolve"

    Raises:
        ValueError: اگر نام استراتژی معتبر نباشد.
    """
    if strategy_name not in VALID_STRATEGIES:
        raise ValueError(
            f"strategy_name must be one of {VALID_STRATEGIES}, got '{strategy_name}'"
        )

    # import داخل تابع برای multiprocessing.spawn safety
    from src.abm import dispatcher_agent
    from src.baselines.random_baseline import random_dispatch_step
    from src.baselines.greedy_baseline import greedy_dispatch_step
    from src.baselines.hungarian_baseline import hungarian_dispatch_step
    from src.baselines.hungarian_resolve_baseline import hungarian_resolve_dispatch_step

    strategy_map = {
        "random": random_dispatch_step,
        "greedy": greedy_dispatch_step,
        "hungarian": hungarian_dispatch_step,
        "hungarian_resolve": hungarian_resolve_dispatch_step,
    }

    saved = dispatcher_agent.DispatcherAgent.dispatch_step
    dispatcher_agent.DispatcherAgent.dispatch_step = strategy_map[strategy_name]
    logger.debug("dispatcher_strategy: patched dispatch_step → %s", strategy_name)
    try:
        yield
    finally:
        dispatcher_agent.DispatcherAgent.dispatch_step = saved
        logger.debug("dispatcher_strategy: restored original dispatch_step")
