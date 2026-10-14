
from .geometry import AA_GEOMETRY

# Standard tokenization for Omniprot and Omniprot-interacting models
OMNIPROT_TOKENS = "ABCDEFGHIKLMNOPQRSTUVWYXZ*-#"
POTTS_EXTENDED_TOKENS = "ACDEFGHIKLMNPQRSTVWY-*#"
PAD = "-"
START = "@"
STOP = "*"
MASK = "#"
DNA_TOKENS = "ACGT"
RNA_TOKENS = "AGCU"
PROTEIN_TOKENS = "ACDEFGHIKLMNPQRSTVWY"

# Minimal 20-letter alphabet and corresponding triplet codes
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA20_1_TO_3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
AA20_3 = [AA20_1_TO_3[aa] for aa in AA20]

# Adding noncanonical amino acids
NONCANON_AA = [
    "HSD",
    "HSE",
    "HSC",
    "HSP",
    "MSE",
    "CSO",
    "SEC",
    "CSX",
    "HIP",
    "SEP",
    "TPO",
]
AA31_3 = AA20_3 + NONCANON_AA

# Chain alphabet for PDB chain naming
CHAIN_ALPHABET = "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# Standard atom indexing
ATOMS_BB = ["N", "CA", "C", "O"]

ATOM_SYMMETRIES = {
    "ARG": [("NH1", "NH2")],  # Correct handling of NH1 and NH2 is relabeling
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
}

AA20_NUM_ATOMS = [4 + len(AA_GEOMETRY[aa]["atoms"]) for aa in AA20_3]
AA20_NUM_CHI = [len(AA_GEOMETRY[aa]["chi_indices"]) for aa in AA20_3]
