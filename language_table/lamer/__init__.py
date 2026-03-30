"""Language Table integration for the LaMer RL training framework."""

__all__ = ["LanguageTableWorker", "LanguageTableMultiProcessEnv"]


def __getattr__(name):
    if name in __all__:
        from .envs import LanguageTableMultiProcessEnv, LanguageTableWorker

        exports = {
            "LanguageTableWorker": LanguageTableWorker,
            "LanguageTableMultiProcessEnv": LanguageTableMultiProcessEnv,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
