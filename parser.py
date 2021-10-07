import io
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, 'py-wdf-reader')

from renishawWiRE import WDFReader
from pathlib import Path


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
