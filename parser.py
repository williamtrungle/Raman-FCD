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


def readme(file='README.md'):
    with open("README.md", "r") as readme:
        title = readme.readline().strip('#').strip()
        body = readme.read()
    return title, body


def metadata(file='metadata/molecular-signature2.csv'):
    df = pd.read_csv(file)
    sig = []
    for _, (start, stop, content) in df.iterrows():
        for i in range(int(start), int(stop+1)):
            sig.append({'Wavelength': i, 'Content': content})
    sig = pd.DataFrame(sig).set_index('Wavelength')['Content']
    return sig


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


def raman_shift_peak(spectrum, shift_min=None, shift_max=None):
    """Find the Raman shift where a peak occurs and its count
    
    This function searches the counts in the `y`list/array for
    the maximum count that occurs, limited by the shift min and
    max parameters. The results is both the Raman shift where
    the peak occured, as well as the count value of that peak.
    Care must be taken when selecting the ranges, as a too small
    range might miss the peak, while a too large range might
    include a different peak in the calculation, thus overshadowing
    the intended target.
    
    To find the maximum peak in the entire Raman spectrum, omit both
    shift arguments.
    
    Parameters
    ----------
    spectrum: pd.Series
        The absorption with respect to the wavelengths in nanometer (nm).
    shift_min: int
        The minimum range to start checking, in nanometer (nm).
        Defaults to 0 if unset, and will search from the start
        of the spectrum.
    shift_max: int
        The maximum range to start checking, in nanometer (nm).
        Defaults to None if unset, and will search all the way
        to the end of the spectrum.
    
    Returns
    -------
    wavelength: int
        Where the peak occured, in nanometer (nm).
    absorption: int
        The photon count of the peak.
    
    Example 1
    ---------
    shift_min, shift_max = 1420, 1480
    wavelength, absorption = raman_shift_peak(SPECTRUM, shift_min=shift_min, shift_max=shift_max)
    print(f"The peak between {shift_min} and {shift_max} occured at {wavelength}nm with value {absorption}")
    
    Example 2
    ---------
    shift_min, shift_max = 0, None
    wavelength, absorption = raman_shift_peak(SPECTRUM, shift_min=shift_min, shift_max=shift_max)
    print(f"The highest peak occured at {wavelength}nm with value {absorption}")
    """
    wavelengths = spectrum.index.to_numpy()
    absorptions = spectrum.to_numpy()
    if shift_min is None:
        shift_min = wavelengths.min()
    if shift_max is None:
        shift_max = wavelengths.max()
    x_min = np.where(wavelengths == shift_min)[0][0]
    x_max = np.where(wavelengths == shift_max)[0][0]
    absorption = max(absorptions[x_min:x_max+1])
    index = np.where(absorptions == absorption)[0][0]
    wavelength = wavelengths[index]
    return wavelength, absorption
