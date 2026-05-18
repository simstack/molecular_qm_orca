import re
from pprint import pprint
from typing import Tuple, Optional
from simstack.models.simple_table import SimpleTable


def parse_mayer_analysis(content: str) -> Tuple[Optional[SimpleTable], Optional[SimpleTable]]:
    """
    Parse Mayer population analysis from ORCA output content.

    Args:
        content: Content of the ORCA output file

    Returns:
        Tuple of (mayer_analysis_table, mayer_bond_orders_table)
    """

    # Find the Mayer population analysis section
    mayer_section_match = re.search(
        r'\*\s*MAYER POPULATION ANALYSIS\s*\*.*?'
        r'ATOM\s+NA\s+ZA\s+QA\s+VA\s+BVA\s+FA\s*\n'
        r'(.*?)'
        r'Mayer bond orders larger than',
        content,
        re.DOTALL
    )

    if not mayer_section_match:
        return None, None

    # Parse the main analysis table
    mayer_analysis_table = _parse_mayer_analysis_table(mayer_section_match.group(1))

    # Find the bond orders section
    bond_orders_match = re.search(
        r'Mayer bond orders larger than [0-9.]+\s*\n'
        r'(.*?)(?:\n\s*\n|\Z)',
        content,
        re.DOTALL
    )

    mayer_bond_orders_table = None
    if bond_orders_match:
        mayer_bond_orders_table = _parse_bond_orders_table(bond_orders_match.group(1))

    return mayer_analysis_table, mayer_bond_orders_table


def _parse_mayer_analysis_table(table_content: str) -> Optional[SimpleTable]:
    """Parse the main Mayer analysis table with atomic properties."""

    # Initialize table
    mayer_table = SimpleTable(name="mayer analysis")
    mayer_table.add_column("atom", "str")
    mayer_table.add_column("na", "float")
    mayer_table.add_column("za", "float")
    mayer_table.add_column("qa", "float")
    mayer_table.add_column("va", "float")
    mayer_table.add_column("bva", "float")
    mayer_table.add_column("fa", "float")

    # Parse each line of the table
    lines = table_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match pattern: index element value1 value2 value3 value4 value5 value6
        match = re.match(r'(\d+)\s+([A-Z][a-z]?)\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)', line)
        if match:
            index, element, na, za, qa, va, bva, fa = match.groups()

            row_data = {
                "atom": f"{index} {element}",
                "na": float(na),
                "za": float(za),
                "qa": float(qa),
                "va": float(va),
                "bva": float(bva),
                "fa": float(fa)
            }

            mayer_table.add_row(row_data)

    return mayer_table if len(mayer_table.row) > 0 else None


def _parse_bond_orders_table(bond_content: str) -> Optional[SimpleTable]:
    """Parse the Mayer bond orders section."""

    # Initialize table
    bond_table = SimpleTable(name="mayer bond orders")
    bond_table.add_column("atom1", "str")
    bond_table.add_column("atom2", "str")
    bond_table.add_column("bond_order", "float")

    # Pattern to match bond entries like: B(  0-C ,  1-C ) :   1.1196
    bond_pattern = r'B\(\s*(\d+)-([A-Z][a-z]?)\s*,\s*(\d+)-([A-Z][a-z]?)\s*\)\s*:\s*([0-9.]+)'

    # Find all bond order entries
    matches = re.findall(bond_pattern, bond_content)

    for match in matches:
        atom1_idx, atom1_element, atom2_idx, atom2_element, bond_order = match

        row_data = {
            "atom1": f"{atom1_idx} {atom1_element}",
            "atom2": f"{atom2_idx} {atom2_element}",
            "bond_order": float(bond_order)
        }

        bond_table.add_row(row_data)

    return bond_table if len(bond_table.row) > 0 else None


if __name__ == "__main__":
    pathname = "C:/Users/wolfg/PycharmProjects/simstack-model/applications/electronic_structure/orca/test_results/excited_states_parsing/orca.out"
    with open(pathname, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    table1, table2 = parse_mayer_analysis(content)
    pprint(table1.model_dump(), indent=4)
    pprint(table2.model_dump(), indent=4)