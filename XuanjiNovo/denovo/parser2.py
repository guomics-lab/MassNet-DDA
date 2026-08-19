"""Mass spectrometry data parsers"""
import logging
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from pyteomics.mgf import MGF
import pandas as pd


LOGGER = logging.getLogger(__name__)


class BaseParser(ABC):
    """A base parser class to inherit from.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    id_type : str, optional
        The Hupo-PSI prefix for the spectrum identifier.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level,
        valid_charge=None,
        id_type="scan",
    ):
        """Initialize the BaseParser"""
        self.path = Path(ms_data_file)
        self.ms_level = ms_level
        self.valid_charge = None if valid_charge is None else set(valid_charge)
        self.id_type = id_type
        self.offset = None
        self.precursor_mz = []
        self.precursor_charge = []
        self.scan_id = []
        self.mz_arrays = []
        self.intensity_arrays = []

    @abstractmethod
    def open(self):
        """Open the file as an iterable"""
        pass

    @abstractmethod
    def parse_spectrum(self, spectrum):
        """Parse a single spectrum

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in a given format.
        """
        pass

    def read(self):
        """Read the ms data file"""
        n_skipped = 0
        with self.open() as spectra:
            for spectrum in tqdm(spectra, desc=str(self.path), unit="spectra"):
                try:
                    self.parse_spectrum(spectrum)
                except (IndexError, KeyError, ValueError):
                    n_skipped += 1

        if n_skipped:
            LOGGER.warning(
                "Skipped %d spectra with invalid precursor info", n_skipped
            )

        #self.precursor_mz = np.array(self.precursor_mz, dtype=np.float32)
        #self.precursor_charge = np.array(self.precursor_charge,dtype=np.float32,)

        self.scan_id = np.array(self.scan_id)

        # Build the index
        sizes = np.array([0] + [len(s) for s in self.mz_arrays])
        self.offset = sizes[:-1].cumsum()
        #self.mz_arrays = np.array(self.mz_arrays, dtype = np.float32)
        #self.intensity_arrays = np.array(self.intensity_arrays, dtype=np.float32)
        #self.mz_arrays = np.concatenate(self.mz_arrays).astype(np.float64)
        #self.intensity_arrays = np.concatenate(self.intensity_arrays).astype(np.float32)

    @property
    def n_spectra(self):
        
        """The number of spectra"""
        
        return self.offset.shape[0]

    @property
    def n_peaks(self):
        #might not correct
        """The number of peaks in the file."""
        mz_arrays_temp = np.concatenate(self.mz_arrays)
        return mz_arrays_temp.shape[0]
    


class MzmlParser(BaseParser):
    """Parse mass spectra from an mzML file.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    """

    def __init__(self, ms_data_file, ms_level=2, valid_charge=None):
        """Initialize the MzmlParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
        )

    def open(self):
        """Open the mzML file for reading"""
        return MzML(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in mzML format.
        """
        if spectrum["ms level"] != self.ms_level:
            return

        if self.ms_level > 1:
            precursor = spectrum["precursorList"]["precursor"][0]
            precursor_ion = precursor["selectedIonList"]["selectedIon"][0]
            precursor_mz = float(precursor_ion["selected ion m/z"])
            if "charge state" in precursor_ion:
                precursor_charge = int(precursor_ion["charge state"])
            elif "possible charge state" in precursor_ion:
                precursor_charge = int(precursor_ion["possible charge state"])
            else:
                precursor_charge = 0
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(list(spectrum["m/z array"]))
            self.intensity_arrays.append(list(spectrum["intensity array"]))
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(_parse_scan_id(spectrum["id"]))


class MzxmlParser(BaseParser):
    """Parse mass spectra from an mzXML file.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzXML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    """

    def __init__(self, ms_data_file, ms_level=2, valid_charge=None):
        """Initialize the MzxmlParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
        )
        

    def open(self):
        """Open the mzXML file for reading"""
        return MzXML(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in mzXML format.
        """
        if spectrum["msLevel"] != self.ms_level:
            return

        if self.ms_level > 1:
            precursor = spectrum["precursorMz"][0]
            precursor_mz = float(precursor["precursorMz"])
            precursor_charge = int(precursor.get("precursorCharge", 0))
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(list(spectrum["m/z array"]))
            self.intensity_arrays.append(list(spectrum["intensity array"]))
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(_parse_scan_id(spectrum["id"]))


class MgfParser(BaseParser):
    """Parse mass spectra from an MGF file.

    Parameters
    ----------
    ms_data_file : str or Path
        The MGF file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    annotations : bool
        Include peptide annotations.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level=2,
        valid_charge=None,
        annotationsLabel=False,
    ):
        """Initialize the MgfParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
            id_type="index",
        )
        self.annotationsLabel = annotationsLabel
        self.annotations = []
        self._counter = 0

    def open(self):
        """Open the MGF file for reading"""
        return MGF(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in MGF format.
        """
        if self.ms_level > 1:
            precursor_mz = float(spectrum["params"]["pepmass"][0])
            precursor_charge = int(spectrum["params"].get("charge", [0])[0])
        else:
            precursor_mz, precursor_charge = None, 0

        if self.annotationsLabel:
            self.annotations.append(spectrum["params"].get("seq"))
        else:
            self.annotations.append(spectrum["params"]["title"])

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(list(spectrum["m/z array"]))
            self.intensity_arrays.append(list(spectrum["intensity array"]))
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(self._counter)

        self._counter += 1


class ParquetParser(BaseParser):
    """Parse mass spectra from a Parquet file.

    Parameters
    ----------
    ms_data_file : str or Path
        The Parquet file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    annotationsLabel : bool, optional
        If True, use "peptide" field for annotations. If False, use "scan" field.
    """

    def __init__(self, ms_data_file, ms_level=2, valid_charge=None, annotationsLabel=False):
        """Initialize the ParquetParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
            id_type="scan",
        )
        self.annotationsLabel = annotationsLabel
        self.annotations = []
        self._df = None

    def open(self):
        """Open the Parquet file for reading"""
        self._df = pd.read_parquet(str(self.path))
        if 'precursor_charge' not in self._df.columns:
            self._df['precursor_charge'] = self._df['charge']
        if 'peptide' not in self._df.columns:
            self._df['peptide'] = self._df['precursor_sequence']

        if isinstance(self._df['peptide'][0], (list, np.ndarray)):
            self._df = self._df[['scan', 'peptide', 'mz_array', 'intensity_array', 'precursor_mz', 'precursor_charge', 'sage_discriminant_score']]
            # if is sage result file, explode
            sage_exploeded = self._df.explode(['peptide', 'precursor_charge', 'sage_discriminant_score'])
            sage_exploeded = sage_exploeded.sort_values(by=['scan', 'sage_discriminant_score'],
                                                   ascending=[True, False]).drop_duplicates(subset=['scan'],
                                                                                            keep='first')
            self._df = sage_exploeded[['scan', 'peptide', 'mz_array', 'intensity_array', 'precursor_mz', 'precursor_charge']]
        else:
            self._df = self._df[['scan', 'peptide', 'mz_array', 'intensity_array', 'precursor_mz', 'precursor_charge']]
        return _ParquetContextManager(self._df)

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict or pd.Series
            The dictionary or Series defining the spectrum in Parquet format.
        """
        # Convert Series to dict if needed
        if isinstance(spectrum, pd.Series):
            spectrum = spectrum.to_dict()

        # Extract precursor information
        precursor_mz = float(spectrum.get("precursor_mz", 0.0))
        precursor_charge = int(spectrum.get("precursor_charge", 0))
        
        # Extract m/z and intensity arrays
        mz_array = spectrum.get("mz_array", [])
        intensity_array = spectrum.get("intensity_array", [])
        
        # Convert to lists if they are numpy arrays or other types
        if isinstance(mz_array, np.ndarray):
            mz_array = mz_array.tolist()
        elif not isinstance(mz_array, list):
            mz_array = list(mz_array) if mz_array is not None else []
            
        if isinstance(intensity_array, np.ndarray):
            intensity_array = intensity_array.tolist()
        elif not isinstance(intensity_array, list):
            intensity_array = list(intensity_array) if intensity_array is not None else []

        # Extract scan ID
        scan_id = spectrum.get("scan", None)
        if scan_id is None:
            # Try to parse from index if scan is not available
            scan_id = 0
        else:
            try:
                scan_id = int(scan_id)
            except (ValueError, TypeError):
                scan_id = _parse_scan_id(str(scan_id)) if isinstance(scan_id, str) else 0

        # Extract annotation (peptide or scan)
        if self.annotationsLabel:
            peptide = spectrum.get("peptide")
            peptide_temp = peptide
            if peptide[1] == '.':
                peptide = peptide[2:-2]
            peptide = peptide.replace("C[57.0215]", "C+57.021").replace("M[15.9949]", "M+15.995").replace("n[42.0106]", "+42.011").replace("[43.0058]", "+43.006").replace("[17.0265]", "-17.027")
            peptide = peptide.replace("C[57.02]", "C+57.021").replace("M[15.99]", "M+15.995").replace("n[42.01]",
                                                                                                      "+42.011").replace(
                "n[42]", "+42.011")
            #print("/n","before peptide: ", peptide_temp, "after peptide: ", peptide, "/n")
            
            # if peptide and isinstance(peptide, str):
            #     # Remove prefix (X.) and suffix (.Y) from peptide like "R.GGGFGGGSSFGGGSGFSGGGFGGGGFGGGR.F"
            #     # Pattern: single_char.sequence.single_char -> extract middle sequence
            #     parts = peptide.split(".", 2)
            #     if len(parts) == 3:
            #         # Extract middle part (the actual sequence)
            #         peptide = parts[1]
            self.annotations.append(peptide)
        else:
            #self.annotations.append(spectrum.get("peptide"))
            self.annotations.append(str(spectrum.get("scan")))

        # Filter by charge if specified
        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(mz_array)
            self.intensity_arrays.append(intensity_array)
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(scan_id)


class _ParquetContextManager:
    """Context manager for iterating over parquet dataframe rows."""
    
    def __init__(self, df):
        self.df = df
    
    def __enter__(self):
        return (row.to_dict() for _, row in self.df.iterrows())
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _parse_scan_id(scan_str):
    """Remove the string prefix from the scan ID.

    Adapted from:
    https://github.com/bittremieux/GLEAMS/blob/
    8831ad6b7a5fc391f8d3b79dec976b51a2279306/gleams/
    ms_io/mzml_io.py#L82-L85

    Parameters
    ----------
    scan_str : str
        The scan ID string.

    Returns
    -------
    int
        The scan ID number.
    """
    try:
        return int(scan_str)
    except ValueError:
        try:
            return int(scan_str[scan_str.find("scan=") + len("scan=") :])
        except ValueError:
            pass

    raise ValueError(f"Failed to parse scan number")
