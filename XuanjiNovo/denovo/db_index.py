import lmdb
import logging
from pathlib import Path
from .record_codec import (
    FORMAT_KEY, FORMAT_VERSION, REBUILD_MESSAGE, decode_record, encode_record,
)

LOGGER = logging.getLogger(__name__)
def listify(obj):
    """Turn an object into a list, but don't split strings."""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)
class DB_Index:
    '''
    read and store and manage (read and write) a signle IMDB file 
    '''
    def __init__(self, db_path, filenames=None, ms_level = 2, valid_charge = None, annotated = True, lock = True ):
        '''
        self.n_spectra is the current index to write 
        e.g, if self.n_spectra = 0, we update(0, spectra)
        '''
        #cant overwrite rn if there is already a db file in path
        index_path = Path(db_path)
        self.db_path = index_path
        self.lock = lock
        
        
        
        self.filenames = filenames
        self.ms_level=ms_level
        self.valid_charge=valid_charge
        self.annotated = bool(annotated)
        is_exist = index_path.exists()
        
        self._init_db()
        
        #create a new 
        if not is_exist:
            txn = self.env.begin(write=True)
            txn.put(FORMAT_KEY, FORMAT_VERSION)
            txn.put( "ms_level".encode(), str(self.ms_level).encode())
            txn.put("n_spectra".encode(),  str(0).encode())
            txn.put("n_peaks".encode(), str(0).encode())
            self.n_spectra = 0
           
            txn.commit()
        else:
            try:
                with self.env.begin() as txn:
                    if txn.get(FORMAT_KEY) != FORMAT_VERSION:
                        raise ValueError(REBUILD_MESSAGE)
                    try:
                        self.n_spectra = int(txn.get(b"n_spectra").decode("ascii"))
                        ms_level_read = int(txn.get(b"ms_level").decode("ascii"))
                    except (AttributeError, UnicodeError, ValueError) as exc:
                        raise ValueError("Invalid index metadata") from exc
                    if self.n_spectra < 0 or ms_level_read != self.ms_level:
                        raise ValueError("Invalid or inconsistent index metadata")
            except Exception:
                self.env.close()
                raise
        if filenames is not None:
            filenames = listify(filenames)
            LOGGER.info("Reading %i files...", len(filenames))
            for ms_file in filenames:
                self.add_file(ms_file)
                
                
    def write_to_db(self, parser, n ):
        #write all spectrums in the parser to lmdb
        if n != len(parser.precursor_charge):
            raise ValueError("Spectrum count does not match parser records")
        next_index = self.n_spectra
        with self.env.begin(write=True) as txn:
            for i in range(n):
                collection_data = {
                    "precursor_mz": parser.precursor_mz[i],
                    "precursor_charge": parser.precursor_charge[i],
                    "mz_array": parser.mz_arrays[i],
                    "intensity_array": parser.intensity_arrays[i],
                    "pep": parser.annotations[i],
                }
                txn.put(str(next_index).encode(), encode_record(collection_data))
                next_index += 1
            txn.put(b"n_spectra", str(next_index).encode())
        self.n_spectra = next_index
        
    def __getitem__(self, idx):
        with self.env.begin(buffers=True) as txn:
            buffer = txn.get(str(idx).encode())
            data = decode_record(buffer)
        pep = data["pep"]
        out = (data["mz_array"], data["intensity_array"],  data["precursor_mz"], data["precursor_charge"], pep)
        return out
    def __len__ (self):
        return self.n_spectra
           
    def add_file(self, a_file):
        a_file = Path(a_file)
        parser = self._get_parser(a_file)
        parser.read()
        n_spectra_in_file = parser.n_spectra
        self.write_to_db(parser, n_spectra_in_file)          
            
    def _init_db(self):
        if self.lock == True:
            read = False
        else:
            read = True
        self.env = lmdb.open(str(self.db_path), map_size=4099511627776, subdir=False, readonly = read, lock=self.lock)
        
        
    def _get_parser(self, ms_data_file):
        # Reading an existing data-only index does not require peak-parser or
        # model dependencies. Import them only when building an index.
        from .parser2 import MzmlParser, MzxmlParser, MgfParser, ParquetParser
        #to allow change of annotations
        kw_args = dict(ms_level=self.ms_level, valid_charge=self.valid_charge, annotationsLabel=self.annotated)
        if ms_data_file.suffix.lower() == ".mzml":
            return MzmlParser(ms_data_file, **kw_args)

        if ms_data_file.suffix.lower() == ".mzxml":
            return MzxmlParser(ms_data_file, **kw_args)

        if ms_data_file.suffix.lower() == ".mgf":
            return MgfParser(ms_data_file, **kw_args)
        if ms_data_file.suffix.lower() == ".parquet":
            return ParquetParser(ms_data_file, **kw_args)

        raise ValueError("Only mzML, mzXML, and MGF files are supported.")
