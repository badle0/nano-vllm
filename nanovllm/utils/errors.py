"""Helpers for preserving primary failures during best-effort cleanup."""

import warnings


def record_cleanup_failure(
    primary: BaseException,
    where: str,
    cleanup: BaseException,
) -> None:
    """Attach cleanup context without replacing ``primary``.

    ``BaseException.add_note`` was added in Python 3.11, while nano-vLLM also
    supports Python 3.10. The warning fallback is deliberately non-throwing so
    warning policy cannot change exception ordering.
    """
    try:
        text = (
            f"{where} also failed: {type(cleanup).__name__}: {cleanup}"
        )
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            try:
                add_note(text)
                return
            except BaseException:
                pass
        try:
            warnings.warn(text, RuntimeWarning, stacklevel=2)
        except BaseException:
            pass
    except BaseException:
        # Even hostile exception formatting or warning hooks must not replace
        # the failure that caused cleanup to run.
        pass
