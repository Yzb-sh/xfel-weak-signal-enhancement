"""
PDB数据下载和准备脚本

功能：
1. 下载指定数量的PDB结构文件
2. 验证下载的文件完整性
3. 创建测试数据集目录结构
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.pdb_fetcher import PDBFetcher
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


BACKUP_PDB_IDS = [
    "1CRN",
    "1L2Y",
    "1VII",
    "2F4K",
    "1UBQ",
    "1GB1",
    "1PGB",
    "2JOF",
    "1ENH",
    "1WLA",
    "1AKE",
    "3LYZ",
    "1BPI",
    "1R69",
    "1TSR",
    "2CI2",
    "2GB1",
    "2PTL",
    "3GB1",
    "4PTI",
]


def download_test_pdb_files(
    n_samples: int = 10,
    output_dir: Path = None,
    use_backup: bool = True
):
    """
    下载测试用的PDB文件
    
    Args:
        n_samples: 需要下载的样本数量
        output_dir: 输出目录
        use_backup: 是否使用备选PDB ID列表
    """
    if output_dir is None:
        output_dir = project_root / "data" / "test_pdb" / "pdb_raw"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing_files = list(output_dir.glob("*.pdb"))
    if len(existing_files) >= n_samples:
        logger.info(f"已存在 {len(existing_files)} 个PDB文件，跳过下载")
        return [f for f in existing_files[:n_samples]]
    
    fetcher = PDBFetcher(timeout=60, retry_count=3)
    
    if use_backup:
        pdb_ids = BACKUP_PDB_IDS[:n_samples]
        logger.info(f"使用备选PDB ID列表: {pdb_ids}")
    else:
        logger.info("从RCSB搜索PDB结构...")
        pdb_ids = fetcher.search(
            mw_min=10,
            mw_max=100,
            resolution_max=3.5,
            max_results=n_samples * 2
        )
        
        if not pdb_ids:
            logger.warning("搜索失败，使用备选列表")
            pdb_ids = BACKUP_PDB_IDS[:n_samples]
    
    logger.info(f"开始下载 {len(pdb_ids)} 个PDB文件...")
    
    successful, failed = fetcher.download(
        pdb_ids[:n_samples],
        str(output_dir),
        file_format="pdb",
        resume=True
    )
    
    logger.info(f"下载完成: {len(successful)} 成功, {len(failed)} 失败")
    
    if failed:
        logger.warning(f"失败的PDB ID: {failed}")
    
    pdb_files = [output_dir / f"{pdb_id}.pdb" for pdb_id in successful]
    pdb_files = [f for f in pdb_files if f.exists()]
    
    return pdb_files


def validate_pdb_files(pdb_files: list):
    """验证PDB文件完整性"""
    logger.info(f"\n验证 {len(pdb_files)} 个PDB文件...")
    
    valid_files = []
    for pdb_file in pdb_files:
        try:
            with open(pdb_file, 'r') as f:
                content = f.read()
            
            if 'ATOM' in content or 'HETATM' in content:
                valid_files.append(pdb_file)
                logger.info(f"  ✓ {pdb_file.name}: 有效")
            else:
                logger.warning(f"  ✗ {pdb_file.name}: 无原子数据")
        except Exception as e:
            logger.error(f"  ✗ {pdb_file.name}: 读取错误 - {e}")
    
    logger.info(f"\n有效文件: {len(valid_files)}/{len(pdb_files)}")
    return valid_files


def main():
    print("=" * 60)
    print("PDB数据下载和准备")
    print("=" * 60)
    
    pdb_files = download_test_pdb_files(n_samples=10, use_backup=True)
    
    if pdb_files:
        valid_files = validate_pdb_files(pdb_files)
        
        print("\n" + "=" * 60)
        print("数据准备完成!")
        print(f"有效PDB文件数: {len(valid_files)}")
        print(f"存储位置: {valid_files[0].parent if valid_files else 'N/A'}")
        print("=" * 60)
        
        return valid_files
    
    return []


if __name__ == "__main__":
    main()
