

from setuptools import setup

def main():
    # setup()
    from _version import version
    print(f"{version=}")

if __name__ == "__main__":
    main()