import re
from pprint import pprint
from typing import Tuple, Optional
from simstack.models.simple_table import SimpleTable

# Constants for unit conversions
HARTREE_TO_EV = 27.2113834279
HARTREE_TO_CM = 219474.6313632

def parse_orca_excited_states(content: str) -> Tuple[SimpleTable, SimpleTable]:
    """
    Parse ORCA TD-DFT excited states from output file content using the approach in orca_output.py.

    Parses all excited state energies in eV, cm⁻¹, S² and multiplicity into a SimpleTable.
    Parses the orbital transitions into a second SimpleTable with entries: state, orbital1, orbital2, coefficient, c.

    Args:
        content: Content of the ORCA output file as a string

    Returns:
        (states_table, transitions_table)
    """

    # Initialize tables
    states_table = SimpleTable(name="Excited States")
    states_table.add_column("state", "int")
    states_table.add_column("energy_au", "float")
    states_table.add_column("energy_ev", "float")
    states_table.add_column("energy_cm", "float")
    states_table.add_column("s2", "float")
    states_table.add_column("mult", "int")

    transitions_table = SimpleTable(name="Orbital Transitions")
    transitions_table.add_column("state", "int")
    transitions_table.add_column("orbital1", "str")
    transitions_table.add_column("orbital2", "str")
    transitions_table.add_column("coefficient", "float")
    transitions_table.add_column("c", "float")  # c coefficient from parentheses

    # Regular expressions for parsing
    state_header_re = re.compile(
        r'STATE\s+(\d+):\s+E=\s+([\d.-]+)\s+au\s+([\d.]+)\s+eV\s+([\d.]+)\s+cm\*\*-1\s+<S\*\*2>\s+=\s+([\d.]+)\s+Mult\s+(\d+)'
    )

    transition_re = re.compile(
        r'^\s*(\d+[ab]?)\s*->\s*(\d+[ab]?)\s*:\s+([\d.]+)\s+\(c=\s*([-\d.]+)\)'
    )

    started = False
    current_state = None

    lines = content.split('\n')
    for line in lines:
        line = line.strip()

        # Look for the start of TD-DFT section
        if 'TD-DFT' in line and 'EXCITED STATES' in line:
            started = True
            continue

        if not started:
            continue

        # Check if we've reached the end of the excited states section
        if line.startswith('----') and len(line) > 10:
            # This might be the end of the section, but continue to be safe
            continue

        # If we encounter another major section header, stop
        if (line.startswith('***') or
                ('ABSORPTION SPECTRUM' in line) or
                ('TIMINGS' in line) or
                ('TOTAL RUN TIME' in line)):
            break

        # Parse state header
        state_match = state_header_re.search(line)
        if state_match:
            state_num = int(state_match.group(1))
            energy_au = float(state_match.group(2))
            energy_ev = float(state_match.group(3))
            energy_cm = float(state_match.group(4))
            s2 = float(state_match.group(5))
            mult = int(state_match.group(6))

            current_state = state_num

            states_table.add_row({
                "state": state_num,
                "energy_au": energy_au,
                "energy_ev": energy_ev,
                "energy_cm": energy_cm,
                "s2": s2,
                "mult": mult
            })
            continue

        # Parse orbital transitions
        if current_state is not None:
            transition_match = transition_re.search(line)
            if transition_match:
                orbital1 = transition_match.group(1)
                orbital2 = transition_match.group(2)
                coefficient = float(transition_match.group(3))
                c = float(transition_match.group(4))

                transitions_table.add_row({
                    "state": current_state,
                    "orbital1": orbital1,
                    "orbital2": orbital2,
                    "coefficient": coefficient,
                    "c": c
                })

    return states_table, transitions_table

if __name__ == "__main__":
    # Example usage - read file and pass content to the function
    with open("C:/Users/wolfg/PycharmProjects/simstack-model/applications/electronic_structure/orca/test_results/excited_states_parsing/orca.out", 'r', encoding='utf-8', errors='ignore') as f:
        file_content = f.read()

    states_table, transitions_table = parse_orca_excited_states(file_content)
    pprint(states_table.model_dump(), indent=4)
    pprint(transitions_table.model_dump(), indent=4)
