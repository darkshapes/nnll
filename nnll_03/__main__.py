### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


if __name__ == "__main__":
    import asyncio
    from nnll_03 import bulk_download

    remote_file_segments = "test_dl.txt"
    remote_url = "https://alphafold.ebi.ac.uk/files/"
    file_suffix = ".pdb"
    local_download_folder = "/home/maxtretikov/Archive/Code/genie2/data/afdbreps_l-256_plddt_80/pdbs"
    print("\n\n\n\n")
    asyncio.run(
        bulk_download(
            remote_file_segments,
            remote_url,
            file_suffix,
            local_download_folder,
        )
    )
