
import numpy as np
import torch
import torch.nn.functional as F


def normed_vec(V: torch.Tensor, distance_eps: float = 1e-3) -> torch.Tensor:
    # Unit vector from i to j
    mag_sq = (V ** 2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + distance_eps)
    U = V / mag
    return U


def normed_cross(
    V1: torch.Tensor, V2: torch.Tensor, distance_eps: float = 1e-3
) -> torch.Tensor:
    C = normed_vec(torch.cross(V1, V2, dim=-1), distance_eps=distance_eps)
    return C


def lengths(
    atom_i: torch.Tensor, atom_j: torch.Tensor, distance_eps: float = 1e-3
) -> torch.Tensor:
    # Bond length of i-j
    dX = atom_j - atom_i
    L = torch.sqrt((dX ** 2).sum(dim=-1) + distance_eps)
    return L


def angles(
    atom_i: torch.Tensor,
    atom_j: torch.Tensor,
    atom_k: torch.Tensor,
    distance_eps: float = 1e-3,
    degrees: bool = False,
) -> torch.Tensor:

    # Bond angle of i-j-k
    U_ji = normed_vec(atom_i - atom_j, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    inner_prod = torch.einsum("bix,bix->bi", U_ji, U_jk)
    inner_prod = torch.clamp(inner_prod, -1, 1)
    A = torch.acos(inner_prod)
    if degrees:
        A = A * 180.0 / np.pi
    return A


def dihedrals(
    atom_i: torch.Tensor,
    atom_j: torch.Tensor,
    atom_k: torch.Tensor,
    atom_l: torch.Tensor,
    distance_eps: float = 1e-3,
    degrees: bool = False,
) -> torch.Tensor:
    U_ij = normed_vec(atom_j - atom_i, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    U_kl = normed_vec(atom_l - atom_k, distance_eps=distance_eps)
    normal_ijk = normed_cross(U_ij, U_jk, distance_eps=distance_eps)
    normal_jkl = normed_cross(U_jk, U_kl, distance_eps=distance_eps)
    # _inner_product = lambda a, b: torch.einsum("bix,bix->bi", a, b)
    _inner_product = lambda a, b: (a * b).sum(-1)
    cos_dihedrals = _inner_product(normal_ijk, normal_jkl)
    angle_sign = _inner_product(U_ij, normal_jkl)
    cos_dihedrals = torch.clamp(cos_dihedrals, -1, 1)
    D = torch.sign(angle_sign) * torch.acos(cos_dihedrals)
    if degrees:
        D = D * 180.0 / np.pi
    return D


def extend_atoms(
    X_1: torch.Tensor,
    X_2: torch.Tensor,
    X_3: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    dihedrals: torch.Tensor,
    distance_eps: float = 1e-3,
    degrees: bool = False,
) -> torch.Tensor:
    if degrees:
        angles *= np.pi / 180.0
        dihedrals *= np.pi / 180.0

    r_32 = X_2 - X_3
    r_12 = X_2 - X_1
    n_1 = normed_vec(r_32, distance_eps=distance_eps)
    n_2 = normed_cross(n_1, r_12, distance_eps=distance_eps)
    n_3 = normed_cross(n_1, n_2, distance_eps=distance_eps)

    lengths = lengths.unsqueeze(-1)
    cos_angle = torch.cos(angles).unsqueeze(-1)
    sin_angle = torch.sin(angles).unsqueeze(-1)
    cos_dihedral = torch.cos(dihedrals).unsqueeze(-1)
    sin_dihedral = torch.sin(dihedrals).unsqueeze(-1)

    X_4 = X_3 + lengths * (
        cos_angle * n_1
        + (sin_angle * sin_dihedral) * n_2
        + (sin_angle * cos_dihedral) * n_3
    )
    return X_4


def quaternions_from_rotations(R: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:

    batch_dims = list(R.shape)[:-2]
    R_flat = R.reshape(batch_dims + [9])
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R_flat.unbind(-1)

    # Quaternion possesses both an axis and angle of rotation
    _sqrt = lambda r: torch.sqrt(F.relu(r) + eps)
    q_angle = _sqrt(1 + R00 + R11 + R22).unsqueeze(-1)
    magnitudes = _sqrt(
        1 + torch.stack([R00 - R11 - R22, -R00 + R11 - R22, -R00 - R11 + R22], -1)
    )
    signs = torch.sign(torch.stack([R21 - R12, R02 - R20, R10 - R01], -1))
    q_axis = signs * magnitudes

    # Normalize (for safety and a missing factor of 2)
    q_unc = torch.cat((q_angle, q_axis), -1)
    q = normed_vec(q_unc, distance_eps=eps)
    return q


def rotations_from_quaternions(
    q: torch.Tensor, normalize: bool = False, eps: float = 1e-3
) -> torch.Tensor:

    batch_dims = list(q.shape)[:-1]
    if normalize:
        q = normed_vec(q, distance_eps=eps)

    a, b, c, d = q.unbind(-1)
    a2, b2, c2, d2 = a ** 2, b ** 2, c ** 2, d ** 2
    R = torch.stack(
        [
            a2 + b2 - c2 - d2,
            2 * b * c - 2 * a * d,
            2 * b * d + 2 * a * c,
            2 * b * c + 2 * a * d,
            a2 - b2 + c2 - d2,
            2 * c * d - 2 * a * b,
            2 * b * d - 2 * a * c,
            2 * c * d + 2 * a * b,
            a2 - b2 - c2 + d2,
        ],
        dim=-1,
    )

    R = R.view(batch_dims + [3, 3])
    return R


def frames_from_backbone(X: torch.Tensor, distance_eps: float = 1e-3):

    X_N, X_CA, X_C, X_O = X.unbind(-2)
    u_CA_N = normed_vec(X_N - X_CA, distance_eps)
    u_CA_C = normed_vec(X_C - X_CA, distance_eps)
    n_1 = u_CA_N
    n_2 = normed_cross(n_1, u_CA_C, distance_eps)
    n_3 = normed_cross(n_1, n_2, distance_eps)
    R = torch.stack([n_1, n_2, n_3], -1)
    return R, X_CA


def hat(omega: torch.Tensor) -> torch.Tensor:

    target = torch.zeros(*omega.size()[:-1], 9, device=omega.device)
    index1 = torch.tensor([7, 2, 3], device=omega.device).expand(
        *target.size()[:-1], -1
    )
    index2 = torch.tensor([5, 6, 1], device=omega.device).expand(
        *target.size()[:-1], -1
    )
    return (
        target.scatter(-1, index1, omega)
        .scatter(-1, index2, -omega)
        .reshape(*target.size()[:-1], 3, 3)
    )


def V(omega: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    I = torch.eye(3, device=omega.device).expand(*omega.size()[:-1], 3, 3)
    theta = omega.pow(2).sum(dim=-1, keepdim=True).add(eps).sqrt()[..., None]
    omega_hat = hat(omega)
    M1 = ((1 - theta.cos()) / theta.pow(2)) * (omega_hat)
    M2 = ((theta - theta.sin()) / theta.pow(3)) * (omega_hat @ omega_hat)
    return I + M1 + M2
