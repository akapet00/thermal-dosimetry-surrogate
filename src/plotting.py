import contextlib

import seaborn as sns


@contextlib.contextmanager
def set_paper_context(style='ticks',
                      font='serif',
                      font_scale=1.15,
                      **kwargs):
    """Set aspects of the visual theme for seaborn plots.

    Parameters
    ----------
    style : string or dict, optional
        Axes style parameters.
    font : string, optional
        Font family, see matplotlib font manager.
    font_scale: float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    kwargs : dict, optional
        Additional keyword arguments for `seaborn.set_theme`.
    """
    sns.set_theme(style=style, font=font, font_scale=font_scale,
                  rc={'text.usetex': True,
                      'text.latex.preamble': r'\usepackage{amsmath}',
                      'font.family': 'serif'},
                  **kwargs)
    yield


@contextlib.contextmanager
def set_poster_context(style='ticks',
                       font='Helvetica',
                       font_scale=1.75,
                       **kwargs):
    """Set aspects of the visual theme for seaborn plots.

    Parameters
    ----------
    style : string or dict, optional
        Axes style parameters.
    font : string, optional
        Font family, see matplotlib font manager.
    font_scale: float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    kwargs : dict, optional
        Additional keyword arguments for `seaborn.set_theme`.
    """
    sns.set_theme(style=style, font=font, font_scale=font_scale,
                  rc={'text.usetex': True,
                      'text.latex.preamble': r'\usepackage{amsmath}',
                      'font.family': 'sans-serif'},
                  **kwargs)
    yield
