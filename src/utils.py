
import hashlib
import pathlib

def comp_sha256(file_name):
    """
    Compute the SHA256 hash of a file.
    Parameters
    ----------
    file_name : str
        Absolute or relative pathname of the file that shall be parsed.
    Returns
    -------
    sha256_res : str
        Resulting SHA256 hash.
    """
    # Set the SHA256 hashing
    hash_sha256 = hashlib.sha256()

    # Open the file in binary mode (read-only) and parse it in 65,536 byte chunks (in case of
    # large files, the loading will not exceed the usable RAM)
    with pathlib.Path(file_name).open(mode="rb") as f_temp:
        for _seq in iter(lambda: f_temp.read(65536), b""):
            hash_sha256.update(_seq)

    # Digest the SHA256 result
    sha256_res = hash_sha256.hexdigest()

    return sha256_res
