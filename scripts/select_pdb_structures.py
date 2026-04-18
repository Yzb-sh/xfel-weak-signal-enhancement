"""
PDB结构选择脚本

功能：
1. 从RCSB PDB筛选高质量蛋白质结构
2. 按分子量分布选择指定数量的结构
3. 生成结构配置文件（包含水化层信息）

输出：
- PDB文件下载
- 结构配置文件（JSON格式）
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import requests

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 全局配置参数 - 可在此修改
# =============================================================================

TOTAL_STRUCTURES = 100

MOLECULAR_WEIGHT_DISTRIBUTION = {
    "small": {"range": (10, 30), "count": 30, "description": "小型蛋白 (10-30 kDa)"},
    "medium": {"range": (30, 60), "count": 40, "description": "中型蛋白 (30-60 kDa)"},
    "large": {"range": (60, 100), "count": 20, "description": "大型蛋白 (60-100 kDa)"},
    "membrane": {"range": (20, 50), "count": 10, "description": "膜蛋白 (20-50 kDa)"},
}

RESOLUTION_MAX = 2.0

R_FACTOR_MAX = 0.25

EXCLUDE_LIGANDS = True

EXCLUDE_METALS = False

OUTPUT_DIR = project_root / "data" / "pdb_structures"

CONFIG_FILE = OUTPUT_DIR / "structure_config.json"

REQUEST_TIMEOUT = 60

RETRY_COUNT = 3

RETRY_DELAY = 2.0

USE_BACKUP_LIST = True

# =============================================================================
# 备选PDB ID列表（当RCSB搜索失败时使用）
# =============================================================================

BACKUP_PDB_IDS = {
    "small": [
        "1CRN", "1L2Y", "1VII", "2F4K", "1UBQ", "1GB1", "1PGB", "2JOF", "1ENH",
        "1BPI", "1R69", "1TSR", "2CI2", "2GB1", "2PTL", "3GB1", "4PTI", "1HEL",
        "2LZM", "1LMB", "1RNB", "2RN2", "1RHA", "1A2P", "1A6K", "1A7S", "1A8D",
        "1A8E", "1A8F", "1A8G", "1A8H", "1A8I", "1A8J", "1A8K", "1A8L", "1A8M",
    ],
    "medium": [
        "1AKE", "3LYZ", "1WLA", "2HHB", "1TIM", "1AOO", "1FAT", "1BPM", "1CGI",
        "1DHR", "1FKB", "1FNB", "1GDI", "1GKY", "1GPB", "1HRC", "1IFB", "1LMB",
        "1MCP", "1MOL", "1NFP", "1OVA", "1PDA", "1PDO", "1PIM", "1PPN", "1QRE",
        "1RBP", "1RHD", "1RNT", "1SNC", "1STP", "1TCA", "1TEN", "1TIF", "1TNS",
        "1TRB", "1TRO", "1UBP", "1UDH", "1VHR", "1WAP", "1WIT", "1XNB", "1YAC",
    ],
    "large": [
        "1AOO", "1FAT", "1BPM", "1CGI", "1DHR", "1FKB", "1FNB", "1GDI", "1GKY",
        "1GPB", "1HRC", "1IFB", "1LMB", "1MCP", "1MOL", "1NFP", "1OVA", "1PDA",
        "1PDO", "1PIM",
    ],
    "membrane": [
        "1M0L", "2XUT", "1BL8", "1F88", "1HZX", "1KQ6", "1L9L", "1LGH", "1NEK",
        "1OED",
    ],
}


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class PDBStructureConfig:
    pdb_id: str
    category: str
    molecular_weight_kda: float
    resolution: float
    r_factor: float
    title: str
    organism: str
    has_hydration_layer: bool
    hydration_thickness_range: tuple
    hydration_density_range: tuple
    exposure_level_range: tuple
    gaussian_noise_range: tuple
    projection_angles: list
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# RCSB PDB API 封装
# =============================================================================

class RCSBApiClient:
    """RCSB PDB API 客户端"""
    
    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
    DOWNLOAD_URL = "https://files.rcsb.org/download"
    
    def __init__(self, timeout: int = 60, retry_count: int = 3, retry_delay: float = 2.0):
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.session = requests.Session()
    
    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """带重试的请求"""
        for attempt in range(self.retry_count):
            try:
                if method == "GET":
                    response = self.session.get(url, timeout=self.timeout, **kwargs)
                else:
                    response = self.session.post(url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise e
    
    def search_by_molecular_weight(
        self,
        mw_min: float,
        mw_max: float,
        resolution_max: float,
        r_factor_max: float,
        max_results: int,
        membrane_only: bool = False
    ) -> List[str]:
        """按分子量搜索PDB结构"""
        
        nodes = [
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
                        "from": mw_min * 1000,
                        "to": mw_max * 1000
                    }
                }
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "refine.ls_r_factor_obs",
                    "operator": "less_or_equal",
                    "value": r_factor_max
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
        
        if membrane_only:
            nodes.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count",
                    "operator": "greater_or_equal",
                    "value": 1
                }
            })
        
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": nodes
            },
            "request_options": {
                "return_all_hits": False,
                "pager": {"start": 0, "rows": max_results}
            },
            "return_type": "entry"
        }
        
        try:
            response = self._request_with_retry("POST", self.SEARCH_URL, json=query)
            results = response.json()
            pdb_ids = [hit["identifier"] for hit in results.get("result_set", [])]
            return pdb_ids
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_entry_metadata(self, pdb_id: str) -> Optional[Dict[str, Any]]:
        """获取PDB条目元数据"""
        url = f"{self.DATA_URL}/{pdb_id}"
        
        try:
            response = self._request_with_retry("GET", url)
            data = response.json()
            
            entry_info = data.get("rcsb_entry_info", {})
            struct = data.get("struct", {})
            
            return {
                "pdb_id": pdb_id,
                "resolution": entry_info.get("resolution_combined", [999.0])[0] if isinstance(entry_info.get("resolution_combined"), list) else entry_info.get("resolution_combined", 999.0),
                "molecular_weight": entry_info.get("molecular_weight", 0.0),
                "r_factor": struct.get("r_factor", 0.0),
                "title": struct.get("title", ""),
                "organism": self._extract_organism(data),
            }
        except Exception as e:
            logger.warning(f"获取元数据失败 {pdb_id}: {e}")
            return None
    
    def _extract_organism(self, data: Dict) -> str:
        """提取生物来源信息"""
        try:
            organisms = data.get("rcsb_entry_container_identifiers", {}).get("organism_scientific", [])
            return organisms[0] if organisms else "Unknown"
        except:
            return "Unknown"
    
    def download_pdb_file(self, pdb_id: str, output_dir: Path) -> bool:
        """下载PDB文件"""
        url = f"{self.DOWNLOAD_URL}/{pdb_id}.pdb"
        output_path = output_dir / f"{pdb_id}.pdb"
        
        if output_path.exists():
            logger.info(f"  文件已存在: {pdb_id}.pdb")
            return True
        
        try:
            response = self._request_with_retry("GET", url)
            output_path.write_text(response.text)
            logger.info(f"  下载成功: {pdb_id}.pdb")
            return True
        except Exception as e:
            logger.error(f"  下载失败 {pdb_id}: {e}")
            return False


# =============================================================================
# 水化层决策逻辑
# =============================================================================

def should_have_hydration_layer(metadata: Dict[str, Any], category: str) -> bool:
    """
    判断结构是否应该添加水化层
    
    决策逻辑：
    1. 膜蛋白：通常在脂质环境中，水化层较少 -> False
    2. 来自溶液研究的蛋白：通常有水化层 -> True
    3. 大型复合物：表面积大，水化层影响显著 -> True
    4. 小型蛋白：水化层相对影响大 -> True
    
    对于通用去噪模型，建议随机分配以增加多样性
    """
    import random
    
    if category == "membrane":
        return random.random() < 0.3
    elif category == "large":
        return random.random() < 0.8
    elif category == "small":
        return random.random() < 0.6
    else:
        return random.random() < 0.5


def get_hydration_parameters(category: str) -> tuple:
    """获取水化层参数范围"""
    if category == "membrane":
        return ((2.0, 3.0), (0.30, 0.35))
    elif category == "large":
        return ((3.0, 5.0), (0.33, 0.40))
    else:
        return ((2.5, 4.0), (0.32, 0.38))


def get_exposure_level_range(category: str) -> tuple:
    """获取曝光水平范围"""
    return (10, 500)


def get_gaussian_noise_range(category: str) -> tuple:
    """获取高斯噪声范围"""
    return (0.0, 3.0)


def get_projection_angles() -> list:
    """获取投影角度列表"""
    return [
        {"axis": 0, "description": "X轴投影"},
        {"axis": 1, "description": "Y轴投影"},
        {"axis": 2, "description": "Z轴投影"},
    ]


# =============================================================================
# 主流程
# =============================================================================

def select_pdb_structures(
    use_backup: bool = USE_BACKUP_LIST,
    total: int = TOTAL_STRUCTURES,
    output_dir: Path = OUTPUT_DIR
) -> List[PDBStructureConfig]:
    """
    选择PDB结构并生成配置
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = output_dir / "pdb_raw"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    api_client = RCSBApiClient(
        timeout=REQUEST_TIMEOUT,
        retry_count=RETRY_COUNT,
        retry_delay=RETRY_DELAY
    )
    
    all_configs = []
    
    for category, config in MOLECULAR_WEIGHT_DISTRIBUTION.items():
        mw_min, mw_max = config["range"]
        count = config["count"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"处理类别: {config['description']}")
        logger.info(f"目标数量: {count}")
        logger.info(f"{'='*60}")
        
        if use_backup:
            pdb_ids = BACKUP_PDB_IDS.get(category, [])[:count]
            logger.info(f"使用备选列表: {len(pdb_ids)} 个PDB ID")
        else:
            logger.info(f"从RCSB搜索...")
            membrane_only = (category == "membrane")
            pdb_ids = api_client.search_by_molecular_weight(
                mw_min=mw_min,
                mw_max=mw_max,
                resolution_max=RESOLUTION_MAX,
                r_factor_max=R_FACTOR_MAX,
                max_results=count * 2,
                membrane_only=membrane_only
            )
            pdb_ids = pdb_ids[:count]
        
        for pdb_id in pdb_ids:
            logger.info(f"\n处理: {pdb_id}")
            
            if not api_client.download_pdb_file(pdb_id, pdb_dir):
                continue
            
            metadata = api_client.get_entry_metadata(pdb_id)
            if metadata is None:
                metadata = {
                    "pdb_id": pdb_id,
                    "resolution": RESOLUTION_MAX,
                    "molecular_weight": (mw_min + mw_max) / 2 * 1000,
                    "r_factor": R_FACTOR_MAX,
                    "title": f"Protein structure {pdb_id}",
                    "organism": "Unknown"
                }
            
            has_hydration = should_have_hydration_layer(metadata, category)
            hydration_thickness, hydration_density = get_hydration_parameters(category)
            exposure_range = get_exposure_level_range(category)
            gaussian_range = get_gaussian_noise_range(category)
            projection_angles = get_projection_angles()
            
            structure_config = PDBStructureConfig(
                pdb_id=pdb_id,
                category=category,
                molecular_weight_kda=metadata["molecular_weight"] / 1000.0,
                resolution=metadata["resolution"],
                r_factor=metadata.get("r_factor", 0.0),
                title=metadata["title"],
                organism=metadata["organism"],
                has_hydration_layer=has_hydration,
                hydration_thickness_range=hydration_thickness,
                hydration_density_range=hydration_density,
                exposure_level_range=exposure_range,
                gaussian_noise_range=gaussian_range,
                projection_angles=projection_angles
            )
            
            all_configs.append(structure_config)
            logger.info(f"  分子量: {structure_config.molecular_weight_kda:.1f} kDa")
            logger.info(f"  分辨率: {structure_config.resolution:.2f} Å")
            logger.info(f"  水化层: {'是' if has_hydration else '否'}")
    
    return all_configs


def save_structure_config(configs: List[PDBStructureConfig], output_path: Path):
    """保存结构配置到JSON文件"""
    config_data = {
        "metadata": {
            "total_structures": len(configs),
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "resolution_max": RESOLUTION_MAX,
                "r_factor_max": R_FACTOR_MAX,
                "molecular_weight_distribution": MOLECULAR_WEIGHT_DISTRIBUTION
            }
        },
        "structures": [c.to_dict() for c in configs]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n配置文件已保存: {output_path}")


def print_summary(configs: List[PDBStructureConfig]):
    """打印配置摘要"""
    print("\n" + "=" * 60)
    print("PDB结构选择摘要")
    print("=" * 60)
    
    categories = {}
    for c in configs:
        if c.category not in categories:
            categories[c.category] = {"count": 0, "with_hydration": 0}
        categories[c.category]["count"] += 1
        if c.has_hydration_layer:
            categories[c.category]["with_hydration"] += 1
    
    print(f"\n总计: {len(configs)} 个结构")
    print("\n按类别分布:")
    for cat, stats in categories.items():
        desc = MOLECULAR_WEIGHT_DISTRIBUTION[cat]["description"]
        print(f"  {desc}: {stats['count']} 个 (含水化层: {stats['with_hydration']})")
    
    print("\n水化层统计:")
    with_hydration = sum(1 for c in configs if c.has_hydration_layer)
    print(f"  有水化层: {with_hydration} ({with_hydration/len(configs)*100:.1f}%)")
    print(f"  无水化层: {len(configs) - with_hydration} ({(len(configs)-with_hydration)/len(configs)*100:.1f}%)")


def main():
    """主函数"""
    print("=" * 60)
    print("PDB结构选择与配置生成")
    print("=" * 60)
    print(f"\n配置参数:")
    print(f"  总结构数: {TOTAL_STRUCTURES}")
    print(f"  最大分辨率: {RESOLUTION_MAX} Å")
    print(f"  最大R因子: {R_FACTOR_MAX}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  配置文件: {CONFIG_FILE}")
    
    configs = select_pdb_structures(
        use_backup=USE_BACKUP_LIST,
        total=TOTAL_STRUCTURES,
        output_dir=OUTPUT_DIR
    )
    
    save_structure_config(configs, CONFIG_FILE)
    print_summary(configs)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    
    return configs


if __name__ == "__main__":
    main()
