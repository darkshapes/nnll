#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


from setuptools import setup

def main():
    # setup()
    from _version import version
    print(f"{version=}")

if __name__ == "__main__":
    main()