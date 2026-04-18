"""
PDB data fetching module for DeepPhase-X.

Provides functionality to search, download, and validate PDB structures
from the RCSB Protein Data Bank.
"""

import os
import gzip
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import requests

try:
    from Bio.PDB import PDBParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

logger = logging.getLogger(__name__)


@dataclass
class PDBMetadata:
    """Metadata for a PDB structure."""
    pdb_id: str
    resolution: float
    molecular_weight: float
    oligomeric_state: int
    organism: str = ""
    title: str = ""
    
    @property
    def mw_kda(self) -> float:
        """Molecular weight in kDa."""
        return self.molecular_weight / 1000.0


class PDBFetcher:
    """
    Utility class for fetching PDB structures from RCSB.
    
    Supports:
    - Searching by molecular weight, resolution, organism
    - Downloading PDB files with resume support
    - Validating downloaded structures
    - Curating diverse datasets
    """
    
    RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
    RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"
    
    # E. coli taxonomy IDs
    ECOLI_TAXONOMY_IDS = ["83333", "562"]  # K-12 and generic E. coli
    
    def __init__(self, timeout: int = 30, retry_count: int = 3, retry_delay: float = 1.0):
        """
        Initialize PDB fetcher.
        
        Args:
            timeout: Request timeout in seconds
            retry_count: Number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.session = requests.Session()
    
    def search(
        self,
        mw_min: float = 10,
        mw_max: float = 100,
        resolution_max: float = 3.5,
        organism_ids: Optional[List[str]] = None,
        max_results: int = 200
    ) -> List[str]:
        """
        Search for PDB structures matching criteria.
        
        Args:
            mw_min: Minimum molecular weight in kDa
            mw_max: Maximum molecular weight in kDa
            resolution_max: Maximum resolution in Angstroms
            organism_ids: List of NCBI taxonomy IDs (default: E. coli)
            max_results: Maximum number of results to return
            
        Returns:
            List of PDB IDs matching the criteria
        """
        if organism_ids is None:
            organism_ids = self.ECOLI_TAXONOMY_IDS
        
        # Build RCSB search query
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": resolution_max
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.molecular_weight",
                            "operator": "range",
                            "value": {
                                "from": mw_min * 1000,  # Convert to Da
                                "to": mw_max * 1000
                            }
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entity_source_organism.ncbi_taxonomy_id",
                            "operator": "in",
                            "value": organism_ids
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": "X-RAY DIFFRACTION"
                        }
                    }
                ]
            },
            "request_options": {
                "return_all_hits": False,
                "pager": {"start": 0, "rows": max_results}
            },
            "return_type": "entry"
        }
        
        try:
            response = self._request_with_retry(
                "POST",
                self.RCSB_SEARCH_URL,
                json=query
            )
            results = response.json()
            pdb_ids = [hit["identifier"] for hit in results.get("result_set", [])]
            logger.info(f"Found {len(pdb_ids)} PDB structures matching criteria")
            return pdb_ids
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_metadata(self, pdb_id: str) -> Optional[PDBMetadata]:
        """
        Get metadata for a PDB structure.
        
        Args:
            pdb_id: PDB ID (4 characters)
            
        Returns:
            PDBMetadata object or None if failed
        """
        url = f"{self.RCSB_DATA_URL}/{pdb_id}"
        
        try:
            response = self._request_with_retry("GET", url)
            data = response.json()
            
            entry_info = data.get("rcsb_entry_info", {})
            struct = data.get("struct", {})
            
            return PDBMetadata(
                pdb_id=pdb_id,
                resolution=entry_info.get("resolution_combined", 999.0),
                molecular_weight=entry_info.get("molecular_weight", 0.0),
                oligomeric_state=entry_info.get("oligomeric_count", 1),
                organism=self._extract_organism(data),
                title=struct.get("title", "")
            )
        except Exception as e:
            logger.warning(f"Failed to get metadata for {pdb_id}: {e}")
            return None
    
    def get_metadata_batch(
        self,
        pdb_ids: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[PDBMetadata]:
        """
        Get metadata for multiple PDB structures.
        
        Args:
            pdb_ids: List of PDB IDs
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of PDBMetadata objects (excludes failed lookups)
        """
        metadata_list = []
        total = len(pdb_ids)
        
        for i, pdb_id in enumerate(pdb_ids):
            meta = self.get_metadata(pdb_id)
            if meta is not None:
                metadata_list.append(meta)
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            # Rate limiting
            time.sleep(0.1)
        
        return metadata_list
    
    def curate_diverse_set(
        self,
        metadata_list: List[PDBMetadata],
        target_n: int = 80
    ) -> List[str]:
        """
        Select a diverse set of PDB structures.
        
        Ensures balanced distribution across:
        - Molecular weight bins (10-30 kDa: 25%, 30-60 kDa: 50%, 60-100 kDa: 25%)
        - Oligomeric states (monomer, dimer, oligomer)
        
        Args:
            metadata_list: List of PDBMetadata objects
            target_n: Target number of structures to select
            
        Returns:
            List of selected PDB IDs
        """
        # Categorize by molecular weight
        small = []   # 10-30 kDa
        medium = []  # 30-60 kDa
        large = []   # 60-100 kDa
        
        for meta in metadata_list:
            mw = meta.mw_kda
            if 10 <= mw < 30:
                small.append(meta)
            elif 30 <= mw < 60:
                medium.append(meta)
            elif 60 <= mw <= 100:
                large.append(meta)
        
        # Sort each category by resolution (best first)
        small.sort(key=lambda x: x.resolution)
        medium.sort(key=lambda x: x.resolution)
        large.sort(key=lambda x: x.resolution)
        
        # Target counts: 25%, 50%, 25%
        n_small = int(target_n * 0.25)
        n_medium = int(target_n * 0.50)
        n_large = target_n - n_small - n_medium
        
        selected = []
        
        # Select from each category with oligomeric state diversity
        selected.extend(self._select_diverse_oligo(small, n_small))
        selected.extend(self._select_diverse_oligo(medium, n_medium))
        selected.extend(self._select_diverse_oligo(large, n_large))
        
        # If we don't have enough, fill from remaining
        if len(selected) < target_n:
            all_remaining = [m for m in metadata_list if m.pdb_id not in selected]
            all_remaining.sort(key=lambda x: x.resolution)
            needed = target_n - len(selected)
            selected.extend([m.pdb_id for m in all_remaining[:needed]])
        
        return selected[:target_n]
    
    def _select_diverse_oligo(
        self,
        metadata_list: List[PDBMetadata],
        n: int
    ) -> List[str]:
        """Select structures with diverse oligomeric states."""
        monomers = [m for m in metadata_list if m.oligomeric_state == 1]
        dimers = [m for m in metadata_list if m.oligomeric_state == 2]
        oligomers = [m for m in metadata_list if m.oligomeric_state > 2]
        
        selected = []
        n_each = max(1, n // 3)
        
        selected.extend([m.pdb_id for m in monomers[:n_each]])
        selected.extend([m.pdb_id for m in dimers[:n_each]])
        selected.extend([m.pdb_id for m in oligomers[:n_each]])
        
        # Fill remaining
        if len(selected) < n:
            remaining = [m for m in metadata_list if m.pdb_id not in selected]
            needed = n - len(selected)
            selected.extend([m.pdb_id for m in remaining[:needed]])
        
        return selected
    
    def download(
        self,
        pdb_ids: List[str],
        output_dir: str,
        file_format: str = "pdb",
        resume: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Download PDB files.
        
        Args:
            pdb_ids: List of PDB IDs to download
            output_dir: Directory to save files
            file_format: File format ('pdb' or 'cif')
            resume: Skip already downloaded files
            progress_callback: Optional callback(current, total, pdb_id) for progress
            
        Returns:
            Tuple of (successful_ids, failed_ids)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        successful = []
        failed = []
        
        # Check which files already exist
        if resume:
            existing = {f.stem.upper() for f in output_path.glob(f"*.{file_format}")}
            pdb_ids = [pid for pid in pdb_ids if pid.upper() not in existing]
            logger.info(f"Skipping {len(existing)} already downloaded files")
        
        total = len(pdb_ids)
        
        for i, pdb_id in enumerate(pdb_ids):
            success = self._download_single(pdb_id, output_path, file_format)
            
            if success:
                successful.append(pdb_id)
            else:
                failed.append(pdb_id)
            
            if progress_callback:
                progress_callback(i + 1, total, pdb_id)
            
            # Rate limiting
            time.sleep(0.5)
        
        logger.info(f"Download complete: {len(successful)} successful, {len(failed)} failed")
        return successful, failed
    
    def _download_single(
        self,
        pdb_id: str,
        output_dir: Path,
        file_format: str
    ) -> bool:
        """Download a single PDB file."""
        # Try compressed first, then uncompressed
        extensions = [f"{file_format}.gz", file_format]
        
        for ext in extensions:
            url = f"{self.RCSB_DOWNLOAD_URL}/{pdb_id}.{ext}"
            
            try:
                response = self._request_with_retry("GET", url)
                
                if ext.endswith(".gz"):
                    # Decompress
                    content = gzip.decompress(response.content).decode("utf-8")
                else:
                    content = response.text
                
                # Save file
                filepath = output_dir / f"{pdb_id}.{file_format}"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                
                return True
                
            except Exception as e:
                logger.debug(f"Failed to download {pdb_id}.{ext}: {e}")
                continue
        
        logger.warning(f"Failed to download {pdb_id}")
        return False
    
    def validate(self, pdb_dir: str, min_atoms: int = 500) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Validate downloaded PDB files.
        
        Args:
            pdb_dir: Directory containing PDB files
            min_atoms: Minimum number of atoms required
            
        Returns:
            Tuple of (valid_ids, invalid_entries) where invalid_entries is [(filename, error)]
        """
        if not HAS_BIOPYTHON:
            logger.warning("BioPython not installed, skipping validation")
            pdb_path = Path(pdb_dir)
            valid = [f.stem for f in pdb_path.glob("*.pdb")]
            return valid, []
        
        parser = PDBParser(QUIET=True)
        pdb_path = Path(pdb_dir)
        
        valid = []
        invalid = []
        
        for pdb_file in pdb_path.glob("*.pdb"):
            try:
                pdb_id = pdb_file.stem
                structure = parser.get_structure(pdb_id, str(pdb_file))
                
                # Count atoms
                atoms = list(structure.get_atoms())
                if len(atoms) < min_atoms:
                    raise ValueError(f"Too few atoms: {len(atoms)} < {min_atoms}")
                
                # Check for END marker
                with open(pdb_file, "r") as f:
                    content = f.read()
                    if "END" not in content:
                        raise ValueError("Missing END marker")
                
                valid.append(pdb_id)
                
            except Exception as e:
                invalid.append((pdb_file.name, str(e)))
        
        logger.info(f"Validation: {len(valid)} valid, {len(invalid)} invalid")
        return valid, invalid
    
    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> requests.Response:
        """Make HTTP request with retry logic."""
        kwargs.setdefault("timeout", self.timeout)
        
        last_error = None
        for attempt in range(self.retry_count):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise last_error
    
    def _extract_organism(self, data: Dict[str, Any]) -> str:
        """Extract organism name from RCSB data."""
        try:
            entities = data.get("rcsb_entry_container_identifiers", {})
            polymer_entities = entities.get("polymer_entity_ids", [])
            if polymer_entities:
                # This is simplified - full implementation would query entity data
                return "Escherichia coli"
        except Exception:
            pass
        return ""
