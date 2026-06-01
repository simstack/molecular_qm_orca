import pandas as pd
import re

def parse_orbital_energies(filename, is_filename=True, logger=None, parse_last_only=True):
    """
    Parse orbital energies from an ORCA output file into a pandas DataFrame.
    
    Parameters:
    filename (str): Path to the ORCA output file or CONTENT of the file if is_filename=False
    is_filename (bool): If True, treat 'filename' as a file path. If False, treat 'filename' as the content of the file.

    
    Returns:
    pandas.DataFrame: DataFrame containing orbital energies with columns:
                     - orbital_no: Orbital number
                     - occupation: Occupation number (0.0 or 2.0)
                     - energy_hartree: Energy in Hartree
                     - energy_ev: Energy in eV
    """
    if not parse_last_only:
        raise NotImplementedError("Parsing multiple ORBITAL ENERGIES sections is not yet implemented.")
    
    orbital_data = []

    if is_filename:
        with open(filename, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
    else:
        lines = filename.splitlines()

    # Find the start of the orbital energies section (use last occurrence)
    start_indices = []
    for i, line in enumerate(lines):
        if "ORBITAL ENERGIES" in line:
            start_indices.append(i)
    
    start_idx = start_indices[-1] if start_indices else None
    
    if start_idx is None:
        raise ValueError("ORBITAL ENERGIES section not found in the file")
    
    # Skip the header lines and find where the data starts
    data_start = None
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line.startswith("NO   OCC"):
            data_start = i + 1
            break
    
    if data_start is None:
        raise ValueError("Orbital energy data header not found")
    
    # Parse the orbital energy data
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        
        # Stop if we hit a line that doesn't contain orbital data
        if not line or line.startswith("*") or line.startswith("-") or "MOLECULAR ORBITALS" in line:
            break
            
        # Parse the line using regex to handle potential formatting variations
        match = re.match(r'^\s*(\d+)\s+(\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)', line)
        if match:
            if logger is not None:
                logger.info(f"line {line}")
                logger.info(f"match - orbital_no {match.group(1)}")
                logger.info(f"match - occupation {match.group(2)}")
                logger.info(f"match - energy_hartree {match.group(3)}")
                logger.info(f"match - energy_ev {match.group(4)}")
            orbital_no = int(match.group(1))
            
            occupation = float(match.group(2))
            energy_hartree = float(match.group(3))
            energy_ev = float(match.group(4))
            
            if logger is not None:
                logger.info(f"parsed data - orbital_no {orbital_no}")
                logger.info(f"occupation {occupation}")
                logger.info(f"energy_hartree {energy_hartree}")
                logger.info(f"energy_ev {energy_ev}")
            orbital_data.append({
                'orbital_no': orbital_no,
                'occupation': occupation,
                'energy_hartree': energy_hartree,
                'energy_ev': energy_ev
            })
    
    # Create DataFrame
    df = pd.DataFrame(orbital_data)
    
    # Add additional useful columns
    if not df.empty:
        df['orbital_type'] = df['occupation'].apply(lambda x: 'occupied' if x > 0 else 'virtual')
        
        # Find HOMO and LUMO indices
        occupied_orbitals = df[df['occupation'] > 0]
        if not occupied_orbitals.empty:
            homo_idx = occupied_orbitals.index[-1]
            # Initialize with correct dtype to avoid FutureWarning from fillna
            df['is_homo'] = False
            df.loc[homo_idx, 'is_homo'] = True
            
            # LUMO is the orbital immediately after the HOMO (the next orbital in the sequence)
            # This ensures LUMO follows directly after HOMO, regardless of whether there are
            # singly-occupied orbitals in unrestricted calculations
            lumo_idx = homo_idx + 1
            if lumo_idx < len(df):
                df['is_lumo'] = False
                df.loc[lumo_idx, 'is_lumo'] = True
    
    return df

# Example usage with the current file
if __name__ == "__main__":
    # Parse the orbital energies from the current orca.out file
    df_orbitals = parse_orbital_energies("orca.out")
    
    print("Orbital Energies DataFrame:")
    print(df_orbitals.head(15))  # Show first 15 orbitals
    
    print(f"\nTotal number of orbitals: {len(df_orbitals)}")
    print(f"Number of occupied orbitals: {len(df_orbitals[df_orbitals['occupation'] > 0])}")
    print(f"Number of virtual orbitals: {len(df_orbitals[df_orbitals['occupation'] == 0])}")
    
    # Show HOMO and LUMO
    if 'is_homo' in df_orbitals.columns:
        homo = df_orbitals[df_orbitals['is_homo'] == True]
        if not homo.empty:
            print(f"\nHOMO (orbital {homo.iloc[0]['orbital_no']}): {homo.iloc[0]['energy_ev']:.4f} eV")
    
    if 'is_lumo' in df_orbitals.columns:
        lumo = df_orbitals[df_orbitals['is_lumo'] == True]
        if not lumo.empty:
            print(f"LUMO (orbital {lumo.iloc[0]['orbital_no']}): {lumo.iloc[0]['energy_ev']:.4f} eV")
            
            # Calculate HOMO-LUMO gap
            if 'is_homo' in df_orbitals.columns:
                homo = df_orbitals[df_orbitals['is_homo'] == True]
                if not homo.empty:
                    gap = lumo.iloc[0]['energy_ev'] - homo.iloc[0]['energy_ev']
                    print(f"HOMO-LUMO gap: {gap:.4f} eV")
    
    print("\nDataFrame info:")
    print(df_orbitals.info())
