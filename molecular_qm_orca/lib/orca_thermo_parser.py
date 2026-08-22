
import re
from typing import List
from simstack.models.simple_table import SimpleTable


def parse_orca_thermochemistry(content: str) -> List[SimpleTable]:
    """
    Parse ORCA thermochemistry output and return tables as SimpleTable instances.
    
    Args:
        content: The ORCA output file content
        
    Returns:
        List of SimpleTable instances containing parsed data
    """
    tables = []
    
    # Parse vibrational frequencies table
    freq_table = parse_vibrational_frequencies(content)
    if freq_table:
        tables.append(freq_table)
    
    # Parse combined inner energy contributions and corrections
    energy_table = parse_energy_contributions_and_corrections(content)
    if energy_table:
        tables.append(energy_table)
    
    # Parse entropy contributions
    entropy_table = parse_entropy_contributions(content)
    if entropy_table:
        tables.append(entropy_table)
    
    # Parse symmetry number rotational entropy table
    symmetry_table = parse_symmetry_entropy_table(content)
    if symmetry_table:
        tables.append(symmetry_table)
    
    return tables


def parse_vibrational_frequencies(content: str) -> SimpleTable | None:
    """Parse the vibrational frequencies section"""
    freq_pattern = r'freq\.\s+(\d+\.\d+)\s+E\(vib\)\s+\.\.\.\s+(\d+\.\d+)'
    matches = re.findall(freq_pattern, content)
    
    if not matches:
        return None
    
    table = SimpleTable(name="Vibrational Frequencies")
    table.add_column("Frequency (cm⁻¹)", "number")
    table.add_column("E(vib) (Eh)", "number")
    
    for freq, e_vib in matches:
        table.add_row({
            "Frequency (cm⁻¹)": float(freq),
            "E(vib) (Eh)": float(e_vib)
        })
    
    return table


def parse_energy_contributions_and_corrections(content: str) -> SimpleTable:
    """Parse and combine the inner energy contributions and corrections sections"""
    table = SimpleTable(name="Energy Contributions and Corrections")
    table.add_column("Component", "string")
    table.add_column("Energy (Eh)", "number")
    table.add_column("Energy (kcal/mol)", "number")
    
    # Parse Summary of contributions to the inner energy U:
    contributions_section = re.search(
        r'Summary of contributions to the inner energy U:(.*?)-----------------------------------------------------------------------',
        content, re.DOTALL
    )
    
    if contributions_section:
        contribution_lines = contributions_section.group(1).strip().split('\n')
        for line in contribution_lines:
            line = line.strip()
            if not line or line.startswith('Total thermal energy'):
                continue
            
            # Parse lines like: "Electronic energy                ...    -79.33971499 Eh"
            # or "Zero point energy                ...      0.16198632 Eh     101.65 kcal/mol"
            match = re.match(r'([^.]+?)\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh(?:\s+(-?\d+\.\d+)\s+kcal/mol)?', line)
            if match:
                component = match.group(1).strip()
                energy_eh = float(match.group(2))
                energy_kcal = float(match.group(3)) if match.group(3) else None
                
                table.add_row({
                    "Component": component,
                    "Energy (Eh)": energy_eh,
                    "Energy (kcal/mol)": energy_kcal
                })
    
    # Parse Summary of corrections to the electronic energy:
    corrections_section = re.search(
        r'Summary of corrections to the electronic energy:(.*?)-----------------------------------------------------------------------',
        content, re.DOTALL
    )
    
    if corrections_section:
        correction_lines = corrections_section.group(1).strip().split('\n')
        for line in correction_lines:
            line = line.strip()
            if not line or line.startswith('(perhaps') or line.startswith('Total correction'):
                continue
            
            match = re.match(r'([^.]+?)\s+(-?\d+\.\d+)\s+Eh\s+(-?\d+\.\d+)\s+kcal/mol', line)
            if match:
                component = match.group(1).strip()
                energy_eh = float(match.group(2))
                energy_kcal = float(match.group(3))
                
                table.add_row({
                    "Component": component,
                    "Energy (Eh)": energy_eh,
                    "Energy (kcal/mol)": energy_kcal
                })
    
    return table if table.row else None


def parse_entropy_contributions(content: str) -> SimpleTable | None:
    """Parse the entropy contributions section"""
    entropy_section = re.search(
        r'The entropy contributions are T\*S.*?-----------------------------------------------------------------------',
        content, re.DOTALL
    )
    
    if not entropy_section:
        return None
    
    table = SimpleTable(name="Entropy Contributions")
    table.add_column("Component", "string")
    table.add_column("T*S (Eh)", "number")
    table.add_column("T*S (kcal/mol)", "number")
    
    entropy_lines = entropy_section.group(0).split('\n')
    for line in entropy_lines:
        line = line.strip()
        if not line or 'entropy' not in line.lower():
            continue
        
        match = re.match(r'([^.]+?)\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh\s+(-?\d+\.\d+)\s+kcal/mol', line)
        if match:
            component = match.group(1).strip()
            energy_eh = float(match.group(2))
            energy_kcal = float(match.group(3))
            
            table.add_row({
                "Component": component,
                "T*S (Eh)": energy_eh,
                "T*S (kcal/mol)": energy_kcal
            })
    
    return table if table.row else None


def parse_symmetry_entropy_table(content: str) -> SimpleTable | None:
    """Parse the symmetry number rotational entropy table"""
    # Find the symmetry table section
    symmetry_section = re.search(
        r'non-linear molecules -----------------------------------\n(.*?) linear molecules ---------------------------------------',
        content, re.DOTALL
    )
    
    if not symmetry_section:
        return None
    
    table = SimpleTable(name="Rotational Entropy by Symmetry Number")
    table.add_column("Symmetry Number", "number")
    table.add_column("S(rot) (Eh)", "number")
    table.add_column("S(rot) (kcal/mol)", "number")
    
    symmetry_lines = symmetry_section.group(1).split('\n')
    for line in symmetry_lines:
        line = line.strip()
        if not line or not line.startswith('|'):
            continue
        
        # Parse lines like: "|  sn= 1 | S(rot)=       0.00943091 Eh      5.92 kcal/mol|"
        match = re.search(r'sn=\s*(\d+).*?S\(rot\)=\s*(-?\d+\.\d+)\s+Eh\s*(-?\d+\.\d+)\s+kcal/mol', line)
        if match:
            sn = int(match.group(1))
            s_rot_eh = float(match.group(2))
            s_rot_kcal = float(match.group(3))
            
            table.add_row({
                "Symmetry Number": sn,
                "S(rot) (Eh)": s_rot_eh,
                "S(rot) (kcal/mol)": s_rot_kcal
            })
    
    return table if table.row else None


# Example usage
if __name__ == "__main__":
    # Read the ORCA output file
    with open("test_results/thermo_parsing/orca.out", "r", encoding='utf-8', errors='replace') as f:
        test_content = f.read()
    
    # Parse the thermochemistry data
    test_tables = parse_orca_thermochemistry(test_content)
    
    # Print results
    for test_table in test_tables:
        print(f"\n{test_table.name}:")
        print(f"Columns: {test_table.heading}")
        print(f"Rows: {len(test_table.row)}")
        for i, row in enumerate(test_table.row):
            print(f"  Row {i+1}: {row}")
