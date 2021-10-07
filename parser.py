import io
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, 'py-wdf-reader')

from renishawWiRE import WDFReader
from ramanTools import bubblefill, cosmic_rays_removal, SNV, savgol_filter
from pathlib import Path


STEPS = "Cosmic Ray Removal", "Savgol", "Raman", "SNV"


def parse(file):
    if isinstance(file, io.BytesIO):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/file.name
            with path.open('wb') as out:
                out.write(file.read())
            wdf = WDFReader(str(path))
    else:
            wdf = WDFReader(file)
    spectra = np.fliplr(np.reshape(wdf.spectra, (wdf.count, wdf.point_per_spectrum)))
    wavelengths = np.flip(wdf.xdata)
    df = pd.DataFrame(spectra).T
    df.index = np.floor(wavelengths).astype(int)
    df = df.loc[~df.index.duplicated()]
    df = df.reindex(range(df.index.min(), df.index.max()+1))
    df = df.interpolate(method='linear')
    df.index.name = 'wavelength'
    df = df.mean(axis=1)
    return df


def preprocess(df, *steps, window_length=11, polyorder=3, bubblewidths=40):
    if "Cosmic Ray Removal" in steps:
        df = df.apply(cosmic_rays_removal, raw=True)
    if "Savgol" in steps:
        df = df.apply(savgol_filter, raw=True, window_length=window_length, polyorder=polyorder)
    if "Raman" in steps:
        df = df.apply(bubblefill, raw=True, bubblewidths=bubblewidths)
    if "SNV" in steps:
        df = df.apply(SNV, raw=True)
    return df
