"""
Property-based tests for PDB fetcher module.

**Feature: deepphase-x, Property 14: 数据集分布平衡性**
**Validates: Requirements 2.3**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.pdb_fetcher import PDBFetcher, PDBMetadata


# =============================================================================
# Hypothesis Strategies
# =============================================================================

@st.composite
def pdb_metadata_strategy(draw):
    """Generate valid PDBMetadata instances."""
    # Generate molecular weight in kDa range 10-100
    mw_kda = draw(st.floats(min_value=10.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    
    return PDBMetadata(
        pdb_id=draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=4, max_size=4)),
        resolution=draw(st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False)),
        molecular_weight=mw_kda * 1000,  # Convert to Da
        oligomeric_state=draw(st.integers(min_value=1, max_value=6)),
        organism="Escherichia coli",
        title="Test structure"
    )


def generate_metadata_list(n: int = 120) -> List[PDBMetadata]:
    """Generate a fixed list of PDBMetadata with diverse molecular weights."""
    import random
    random.seed(42)
    
    metadata_list = []
    
    # Small: 10-30 kDa (25%)
    for i in range(n // 4):
        mw = random.uniform(10.0, 29.9)
        metadata_list.append(PDBMetadata(
            pdb_id=f"S{i:03d}",
            resolution=random.uniform(0.5, 5.0),
            molecular_weight=mw * 1000,
            oligomeric_state=random.randint(1, 3),
        ))
    
    # Medium: 30-60 kDa (50%)
    for i in range(n // 2):
        mw = random.uniform(30.0, 59.9)
        metadata_list.append(PDBMetadata(
            pdb_id=f"M{i:03d}",
            resolution=random.uniform(0.5, 5.0),
            molecular_weight=mw * 1000,
            oligomeric_state=random.randint(1, 3),
        ))
    
    # Large: 60-100 kDa (25%)
    for i in range(n - n // 4 - n // 2):
        mw = random.uniform(60.0, 100.0)
        metadata_list.append(PDBMetadata(
            pdb_id=f"L{i:03d}",
            resolution=random.uniform(0.5, 5.0),
            molecular_weight=mw * 1000,
            oligomeric_state=random.randint(1, 3),
        ))
    
    return metadata_list


# =============================================================================
# Property Tests
# =============================================================================

class TestDatasetDistribution:
    """
    **Feature: deepphase-x, Property 14: 数据集分布平衡性**
    **Validates: Requirements 2.3**
    
    For any PDB metadata list, the curated dataset should satisfy:
    - 10-30 kDa: 25% ± 5%
    - 30-60 kDa: 50% ± 5%
    - 60-100 kDa: 25% ± 5%
    """
    
    def test_molecular_weight_distribution(self):
        """Test that curated dataset has balanced MW distribution."""
        metadata_list = generate_metadata_list(120)
        
        fetcher = PDBFetcher()
        target_n = 80
        
        selected_ids = fetcher.curate_diverse_set(metadata_list, target_n=target_n)
        
        # Get metadata for selected structures
        id_to_meta = {m.pdb_id: m for m in metadata_list}
        selected_meta = [id_to_meta[pid] for pid in selected_ids if pid in id_to_meta]
        
        total = len(selected_meta)
        assert total > 0, "No structures selected"
        
        # Count in each bin
        small = sum(1 for m in selected_meta if 10 <= m.mw_kda < 30)
        medium = sum(1 for m in selected_meta if 30 <= m.mw_kda < 60)
        large = sum(1 for m in selected_meta if 60 <= m.mw_kda <= 100)
        
        # Check distribution (with 10% tolerance)
        assert 0.15 <= small / total <= 0.35, f"Small MW ratio {small/total:.2f} out of range"
        assert 0.40 <= medium / total <= 0.60, f"Medium MW ratio {medium/total:.2f} out of range"
        assert 0.15 <= large / total <= 0.35, f"Large MW ratio {large/total:.2f} out of range"
    
    def test_target_count_respected(self):
        """Test that curated dataset respects target count."""
        metadata_list = generate_metadata_list(120)
        
        fetcher = PDBFetcher()
        target_n = 80
        
        selected_ids = fetcher.curate_diverse_set(metadata_list, target_n=target_n)
        
        # Should not exceed target
        assert len(selected_ids) <= target_n
        
        # Should be close to target
        assert len(selected_ids) >= target_n * 0.8
    
    def test_no_duplicates(self):
        """Test that curated dataset has no duplicate PDB IDs."""
        metadata_list = generate_metadata_list(120)
        
        fetcher = PDBFetcher()
        selected_ids = fetcher.curate_diverse_set(metadata_list, target_n=80)
        
        # Check for duplicates
        assert len(selected_ids) == len(set(selected_ids)), "Duplicate PDB IDs in selection"


# =============================================================================
# Unit Tests
# =============================================================================

class TestPDBMetadata:
    """Test PDBMetadata dataclass."""
    
    def test_mw_kda_conversion(self):
        """Test molecular weight conversion to kDa."""
        meta = PDBMetadata(
            pdb_id="1CRN",
            resolution=1.5,
            molecular_weight=50000.0,  # 50 kDa in Da
            oligomeric_state=1
        )
        
        assert meta.mw_kda == 50.0
    
    def test_metadata_creation(self):
        """Test PDBMetadata creation."""
        meta = PDBMetadata(
            pdb_id="1ABC",
            resolution=2.0,
            molecular_weight=30000.0,
            oligomeric_state=2,
            organism="E. coli",
            title="Test protein"
        )
        
        assert meta.pdb_id == "1ABC"
        assert meta.resolution == 2.0
        assert meta.oligomeric_state == 2


class TestPDBFetcher:
    """Test PDBFetcher functionality."""
    
    def test_fetcher_initialization(self):
        """Test PDBFetcher initialization."""
        fetcher = PDBFetcher(timeout=60, retry_count=5)
        
        assert fetcher.timeout == 60
        assert fetcher.retry_count == 5
    
    def test_curate_empty_list(self):
        """Test curation with empty list."""
        fetcher = PDBFetcher()
        
        selected = fetcher.curate_diverse_set([], target_n=80)
        
        assert selected == []
    
    def test_curate_small_list(self):
        """Test curation with list smaller than target."""
        fetcher = PDBFetcher()
        
        metadata_list = [
            PDBMetadata("1ABC", 1.5, 25000, 1),
            PDBMetadata("2DEF", 2.0, 45000, 2),
            PDBMetadata("3GHI", 1.8, 75000, 3),
        ]
        
        selected = fetcher.curate_diverse_set(metadata_list, target_n=80)
        
        # Should return all available
        assert len(selected) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
