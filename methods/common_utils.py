"""Common utility functions shared across ET partitioning methods."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


DEFAULT_FLUXNET_PATTERN = re.compile(
    r"^(?:AMF|FLX)_.*_FLUXNET(?:2015)?_FULLSET_\d{4}-\d{4}_\d+-\d+$"
)


def get_expected_csv_filename(folder_name: str) -> str:
    """
    Convert a FLUXNET folder name to its expected CSV filename.
    
    Parameters
    ----------
    folder_name : str
        FLUXNET folder name (e.g., 'FLX_FI-Hyy_FLUXNET2015_FULLSET_2008-2010_1-3')
    
    Returns
    -------
    str
        Expected CSV filename with '_HH_' suffix
        
    Raises
    ------
    ValueError
        If folder name doesn't match expected FLUXNET pattern
    """
    if "_FLUXNET2015_FULLSET_" in folder_name:
        return folder_name.replace(
            "_FLUXNET2015_FULLSET_", "_FLUXNET2015_FULLSET_HH_"
        ) + ".csv"
    if "_FLUXNET_FULLSET_" in folder_name:
        return folder_name.replace(
            "_FLUXNET_FULLSET_", "_FLUXNET_FULLSET_HH_"
        ) + ".csv"
    raise ValueError(f"Folder name does not follow expected pattern: {folder_name}")


def extract_site_id(filename: str) -> str:
    """
    Extract site ID from FLUXNET filename.
    
    Parameters
    ----------
    filename : str
        FLUXNET filename or folder name
        
    Returns
    -------
    str
        Site ID (e.g., 'FI-Hyy')
        
    Raises
    ------
    IndexError
        If filename doesn't contain expected format
    """
    return filename.split("_")[1]


def iter_site_folders(
    base_path: Path, 
    pattern: re.Pattern[str] = DEFAULT_FLUXNET_PATTERN
) -> Iterable[Path]:
    """
    Iterate over FLUXNET site folders matching the specified pattern.
    
    Parameters
    ----------
    base_path : Path
        Directory containing site folders
    pattern : re.Pattern[str], optional
        Regular expression to match folder names
        
    Yields
    ------
    Path
        Path to each matching site folder
        
    Raises
    ------
    FileNotFoundError
        If base_path does not exist
    """
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")
    
    for entry in sorted(base_path.iterdir()):
        if entry.is_dir() and pattern.match(entry.name):
            yield entry


def process_sites_parallel(
    site_items: List,
    process_func: Callable,
    workers: Optional[int] = None,
    description: str = "Processing sites"
) -> Tuple[int, int, List]:
    """
    Process multiple sites in parallel using ProcessPoolExecutor.
    
    Parameters
    ----------
    site_items : List
        List of items to process (folders, files, etc.)
    process_func : Callable
        Function to call for each item. Should return (success: bool, result: Any)
    workers : int, optional
        Number of worker processes. If None, uses CPU count.
    description : str, optional
        Description for progress messages
        
    Returns
    -------
    Tuple[int, int, List]
        (successful_count, failed_count, list_of_errors)
    """
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    print(f"{description} with {workers} workers...")
    
    successful_count = 0
    failed_count = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(process_func, item): item
            for item in site_items
        }
        
        completed = 0
        total = len(site_items)
        
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            completed += 1
            
            try:
                success, result = future.result()
                if success:
                    successful_count += 1
                    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
                else:
                    failed_count += 1
                    error_msg = result if isinstance(result, str) else "Unknown error"
                    errors.append((str(item), error_msg))
                    print(f"[Error processing {item}]: {error_msg}")
            except Exception as exc:
                failed_count += 1
                errors.append((str(item), str(exc)))
                print(f"[Exception processing {item}]: {exc}")
    
    return successful_count, failed_count, errors
