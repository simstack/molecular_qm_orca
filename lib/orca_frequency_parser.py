import re
from typing import List, Optional, Dict, Any, Tuple
from simstack.models.simple_table import SimpleTable


def parse_vibrational_frequencies(content: str) -> Optional[SimpleTable]:
    """
    Parse vibrational frequencies section from ORCA output content.

    Args:
        content: Full ORCA output file content as string

    Returns:
        SimpleTable with frequency data or None if not found
    """
    freq_table = SimpleTable(name="Vibrational Frequencies")
    freq_table.add_column("Mode", "int")
    freq_table.add_column("Frequency (cm⁻¹)", "float")

    # Look for vibrational frequencies section
    freq_section_start = content.find("VIBRATIONAL FREQUENCIES")
    if freq_section_start == -1:
        return None

    # Extract the section and split into lines
    freq_section = content[freq_section_start:]
    lines = freq_section.split('\n')

    # Skip the header lines and find where the data starts
    data_started = False
    for line in lines:
        line = line.strip()

        # Skip the header, empty lines, and scaling factor line
        if (line.startswith('-') or
                line == '' or
                line.startswith('VIBRATIONAL FREQUENCIES') or
                line.startswith('Scaling factor')):
            continue

        # Check if this is a data line with the pattern "     0:       0.00 cm**-1"
        freq_pattern = r'^\s*(\d+):\s*([-]?\d+\.\d+)\s+cm\*\*-1(?:\s+\*\*\*imaginary mode\*\*\*)?'
        match = re.match(freq_pattern, line)

        if match:
            data_started = True
            mode_num = int(match.group(1))
            frequency = float(match.group(2))

            freq_table.add_row({
                "Mode": mode_num,
                "Frequency (cm⁻¹)": frequency,
            })
        elif data_started and line == '':
            # Stop when we hit an empty line after data has started
            break
        elif data_started and not re.match(r'^\s*\d+:', line):
            # Stop if we encounter a line that doesn't match the pattern after data started
            break

    return freq_table if len(freq_table.row) > 0 else None


def parse_normal_modes(content: str) -> Optional[SimpleTable]:
    """
    Parse normal modes section from ORCA output content.
    Handles multiple blocks of normal modes and combines them into a single table.

    Args:
        content: Full ORCA output file content as string

    Returns:
        SimpleTable with normal modes data or None if not found
    """
    modes_table = SimpleTable(name="Normal Modes")

    # Look for normal modes section
    modes_section_start = content.find("NORMAL MODES")
    if modes_section_start == -1:
        return None

    # Extract the normal modes section
    modes_section = content[modes_section_start:]

    # Find the end of the section - look for next major section or large gap
    next_section_patterns = [
        "IR SPECTRUM",
        "RAMAN SPECTRUM",
        "THERMOCHEMISTRY",
        "TIMINGS",
        "TOTAL RUN TIME"
    ]

    modes_section_end = len(modes_section)
    for pattern in next_section_patterns:
        pos = modes_section.find(pattern)
        if pos != -1 and pos < modes_section_end:
            modes_section_end = pos

    modes_section = modes_section[:modes_section_end]
    lines = modes_section.split('\n')

    # Dictionary to store all mode data for each atom
    atom_data = {}
    all_modes = set()  # Keep track of all mode numbers we've seen
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for lines with column numbers like "0          1          2          3          4          5"
        if re.match(r'^\d+\s+\d+\s+\d+', line):
            # Extract mode numbers from header
            mode_numbers = [int(x) for x in line.split()]
            all_modes.update(mode_numbers)
            
            # Parse data rows for this block
            i += 1
            while i < len(lines):
                data_line = lines[i].strip()
                
                # Check if we've hit the end of this block
                if not data_line or data_line.startswith('---'):
                    break
                
                # Check if we've encountered another mode header (start of next block)
                if re.match(r'^\d+\s+\d+\s+\d+', data_line):
                    i -= 1  # Back up one line to process this header in next iteration
                    break
                
                # Check if this is a data row (starts with atom index)
                if re.match(r'^\d+', data_line):
                    values = data_line.split()
                    if len(values) == len(mode_numbers) + 1:  # +1 for atom index
                        try:
                            atom_idx = int(values[0])
                            mode_values = [float(val) for val in values[1:]]
                            
                            # Initialize atom data if not seen before
                            if atom_idx not in atom_data:
                                atom_data[atom_idx] = {}
                            
                            # Store mode values for this atom
                            for j, mode_num in enumerate(mode_numbers):
                                atom_data[atom_idx][mode_num] = mode_values[j]
                                
                        except (ValueError, IndexError):
                            pass  # Skip malformed lines
                
                i += 1
        else:
            i += 1
    
    # If no data was found, return None
    if not atom_data or not all_modes:
        return None
    
    # Create table columns
    modes_table.add_column("Atom_Index", "int")
    for mode_num in sorted(all_modes):
        modes_table.add_column(f"Mode_{mode_num}", "float")
    
    # Add rows to the table
    for atom_idx in sorted(atom_data.keys()):
        row_data = {"Atom_Index": atom_idx}
        
        # Add mode values for this atom (use 0.0 as default for missing modes)
        for mode_num in sorted(all_modes):
            mode_value = atom_data[atom_idx].get(mode_num, 0.0)
            row_data[f"Mode_{mode_num}"] = mode_value
        
        modes_table.add_row(row_data)
    
    return modes_table if len(modes_table.row) > 0 else None


def parse_ir_spectrum(content: str) -> Optional[SimpleTable]:
    """
    Parse IR spectrum section from ORCA output content.

    Args:
        content: Full ORCA output file content as string

    Returns:
        SimpleTable with IR spectrum data or None if not found
    """
    ir_table = SimpleTable(name="IR Spectrum")
    ir_table.add_column("Mode", "int")
    ir_table.add_column("Frequency (cm⁻¹)", "float")
    ir_table.add_column("eps (L/(mol*cm))", "float")
    ir_table.add_column("Intensity (km/mol)", "float")
    ir_table.add_column("T**2 (a.u.)", "float")
    ir_table.add_column("TX", "float")
    ir_table.add_column("TY", "float")
    ir_table.add_column("TZ", "float")

    # Look for IR spectrum section
    ir_section_start = content.find("IR SPECTRUM")
    if ir_section_start == -1:
        return None

    # Extract the IR spectrum section
    ir_section = content[ir_section_start:]
    
    # Find the end of the section - look for next major section or multiple empty lines
    next_section_patterns = [
        "RAMAN SPECTRUM",
        "THERMOCHEMISTRY", 
        "TIMINGS",
        "TOTAL RUN TIME",
        "\n\n\n"  # Multiple empty lines
    ]

    ir_section_end = len(ir_section)
    for pattern in next_section_patterns:
        pos = ir_section.find(pattern)
        if pos != -1 and pos < ir_section_end:
            ir_section_end = pos

    ir_section = ir_section[:ir_section_end]
    lines = ir_section.split('\n')

    # Pattern to match IR spectrum data lines
    # Format: " 13:    609.64   0.000286    1.45  0.000146  ( 0.000008 -0.012101 -0.000010)"
    ir_pattern = r'^\s*(\d+):\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+\(\s*([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s*\)'

    for line in lines:
        line = line.strip()
        match = re.match(ir_pattern, line)
        if match:
            mode_num = int(match.group(1))
            frequency = float(match.group(2))
            eps = float(match.group(3))
            intensity = float(match.group(4))
            t_squared = float(match.group(5))
            tx = float(match.group(6))
            ty = float(match.group(7))
            tz = float(match.group(8))

            ir_table.add_row({
                "Mode": mode_num,
                "Frequency (cm⁻¹)": frequency,
                "eps (L/(mol*cm))": eps,
                "Intensity (km/mol)": intensity,
                "T**2 (a.u.)": t_squared,
                "TX": tx,
                "TY": ty,
                "TZ": tz,
            })

    return ir_table if len(ir_table.row) > 0 else None


def parse_orca_frequencies_file(file_path: str) -> Tuple[Optional[SimpleTable], Optional[SimpleTable], Optional[SimpleTable]]:
    """
    Parse ORCA frequencies output file and extract all frequency-related tables.

    Args:
        file_path: Path to ORCA output file

    Returns:
        Tuple containing (vibrational_frequencies_table, normal_modes_table, ir_spectrum_table)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Parse all three sections using the same content
        vibrational_frequencies = parse_vibrational_frequencies(content)
        normal_modes = parse_normal_modes(content)
        ir_spectrum = parse_ir_spectrum(content)

        return vibrational_frequencies, normal_modes, ir_spectrum

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None, None, None


if __name__ == "__main__":
    # Test the parser
    import os
    file_path = os.path.join("test_results", "frequencies.out")  # Adjust path as needed
    freq_table, modes_table, ir_table = parse_orca_frequencies_file(file_path)

    tables = [
        ("Vibrational Frequencies", freq_table),
        ("Normal Modes", modes_table),
        ("IR Spectrum", ir_table)
    ]

    for name, table in tables:
        if table:
            print(f"\n{name}:")
            print(f"Number of rows: {len(table.row)}")
            print(f"Columns: {table.heading}")
            if len(table.row) > 0:
                print(f"Sample row: {table.row[0]}")
        else:
            print(f"\n{name}: Not found")