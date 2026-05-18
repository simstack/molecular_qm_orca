import re
from pprint import pprint
from typing import Optional
from simstack.models.simple_table import SimpleTable

def parse_orca_absorption_spectrum(content: str) -> Optional[SimpleTable]:
    """
    Parse ORCA absorption spectrum from the output file content using the approach in pyorca.py.

    Parses absorption spectrum data including transition, energy (eV), energy (cm⁻¹),
    wavelength (nm), oscillator strength, D², and dipole moment components (DX, DY, DZ).

    Args:
        content: Content of the ORCA output file as a string

    Returns:
        SimpleTable with absorption spectrum data or None if not found
    """

    # Initialize table
    spectrum_table = SimpleTable(name="Absorption Spectrum")
    spectrum_table.add_column("transition", "str")
    spectrum_table.add_column("energy_ev", "float")
    spectrum_table.add_column("energy_cm", "float")
    spectrum_table.add_column("wavelength_nm", "float")
    spectrum_table.add_column("fosc", "float")
    spectrum_table.add_column("d2", "float")
    spectrum_table.add_column("dx", "float")
    spectrum_table.add_column("dy", "float")
    spectrum_table.add_column("dz", "float")

    # Regular expression for parsing absorption spectrum data
    # Matches lines like: "  0-1A  ->  1-1A    9.931039   80099.2   124.8   0.000000000   0.00000  -0.00000   0.00000  -0.00000"
    spectrum_re = re.compile(
        r'^\s*(\S+\s*->\s*\S+)\s+'  # Transition (e.g., "0-1A -> 1-1A")
        r'([\d.]+)\s+'              # Energy (eV)
        r'([\d.]+)\s+'              # Energy (cm⁻¹)
        r'([\d.]+)\s+'              # Wavelength (nm)
        r'([\d.]+)\s+'              # Oscillator strength fosc(D2)
        r'([\d.]+)\s+'              # D² (au²)
        r'([-\d.]+)\s+'             # DX (au)
        r'([-\d.]+)\s+'             # DY (au)
        r'([-\d.]+)'                # DZ (au)
    )

    started = False

    lines = content.split('\n')
    for line in lines:
        line_stripped = line.strip()

        # Look for the start of absorption spectrum section
        if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
            started = True
            continue

        if not started:
            continue

        # Skip header lines and separators
        if (line_stripped.startswith('---') or
                'Transition' in line_stripped or
                '(eV)' in line_stripped):
            continue

        # Stop parsing when encountering a blank line
        if line_stripped == '':
            break

        # Check if we've reached the end of the spectrum section
        if (line_stripped.startswith('***') or
                'TIMINGS' in line_stripped or
                'TOTAL RUN TIME' in line_stripped or
                line_stripped.startswith('----') and 'ABSORPTION SPECTRUM' not in line):
            break

        # Parse spectrum data
        spectrum_match = spectrum_re.match(line)
        if spectrum_match:
            transition = spectrum_match.group(1).strip()
            energy_ev = float(spectrum_match.group(2))
            energy_cm = float(spectrum_match.group(3))
            wavelength_nm = float(spectrum_match.group(4))
            fosc = float(spectrum_match.group(5))
            d2 = float(spectrum_match.group(6))
            dx = float(spectrum_match.group(7))
            dy = float(spectrum_match.group(8))
            dz = float(spectrum_match.group(9))

            spectrum_table.add_row({
                "transition": transition,
                "energy_ev": energy_ev,
                "energy_cm": energy_cm,
                "wavelength_nm": wavelength_nm,
                "fosc": fosc,
                "d2": d2,
                "dx": dx,
                "dy": dy,
                "dz": dz
            })

    # Return None if no data was found
    if len(spectrum_table.row) == 0:
        return None

    return spectrum_table


if __name__ == "__main__":
    # Example usage - read file and pass content to the function
    pathname = "C:/Users/wolfg/PycharmProjects/simstack-model/applications/electronic_structure/orca/test_results/excited_states_parsing/orca.out"
    with open(pathname, 'r', encoding='utf-8', errors='replace') as f:
        file_content = f.read()

    spectrum = parse_orca_absorption_spectrum(file_content)
    pprint(spectrum.model_dump(), indent=4)