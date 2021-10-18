import re
import subprocess
import sys
from copy import copy
from setuptools_scm import get_version

import os.path

VERSION_STRING = r'(\d+).(\d+).(\d+)'
SETUP_VERSION_PATTERN = re.compile(f'version="({VERSION_STRING})"')
SPECIFIER_PATTERN = re.compile(f'v?{VERSION_STRING}')


class Version:
    def __init__(self, major: int, minor: int, patch: int):
        if not (major >= 0 and minor >= 0 and patch >= 0):
            raise ValueError('All arguments to Version must be positive integers')

        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self):
        return str(self.major) + '.' + str(self.minor) + '.' + str(self.patch)

    @property
    def tag(self):
        return 'v' + str(self)

    def increment_major(self):
        self.major += 1
        self.minor = 0
        self.patch = 0

    def increment_minor(self):
        self.minor += 1
        self.patch = 0

    def increment_patch(self):
        self.patch += 1

    def __eq__(self, other):
        return self.major == other.major and self.minor == other.minor and self.patch == other.patch

    @property
    def tuple(self):
        return self.major, self.minor, self.patch

    @staticmethod
    def from_string(string: str):
        version_match = SPECIFIER_PATTERN.match(string)
        if version_match is None:
            raise ValueError('Version string is not in a valid format')
        version_numbers = [int(v) for v in version_match.groups()[0:3]]
        return Version(*version_numbers)


def matches_start(string: str, pattern: str):
    regex = ''.join([c+'?' for c in pattern])
    return bool(re.fullmatch(regex, string))


def get_current_version():

    def vsch(x):
        return x.tag.base_version

    def ls(x):
        return ""

    cd = os.path.dirname(__file__)

    vs = get_version(cd, version_scheme=vsch, local_scheme=ls)
    return Version.from_string(vs)


def main():  # pragma: no cover
    current_version = get_current_version()
    print(f'Current version is {str(current_version)}')
    try:
        _, version = sys.argv
    except ValueError:
        version = input('Enter a version specifier [vX.Y.Z|major|minor|PATCH]: ')

    try:
        new_version = Version.from_string(version)
    except ValueError:
        new_version = copy(current_version)
        if matches_start(version, 'patch'):
            new_version.increment_patch()
        elif matches_start(version, 'minor'):
            new_version.increment_minor()
        elif matches_start(version, 'major'):
            new_version.increment_major()
        else:
            raise ValueError('Invalid version specifier')

    print(f'New version will be: {str(new_version)}')
    continue_response = input('Continue? [Y/n]: ')
    if continue_response.lower().startswith('n'):
        return
    status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True)
    clean = status.stdout == b''
    do_stash = False
    if not clean:
        subprocess.run(['git', 'status'])
        stash_response = input('Working directory is not clean. '
                               'Do you want to continue with current status, stash, or quit [c/S/q]?: ')
        if stash_response.lower().startswith('q'):
            return
        do_stash = not stash_response.lower().startswith('c')
        if do_stash:
            print('Stashing changes')
            subprocess.run(['git', 'stash'], check=True)
    # tag
    message = input('Enter release message. Leave blank to use the default message):\n')
    if message == '':
        message = f'Release version {str(new_version)}'
    print(f'Using message: {message}')
    subprocess.run(['git', 'tag', '-a', '-m', message, new_version.tag], check=True)
    # push
    push_response = input('Push? [Y/n]: ')
    if not push_response.lower().startswith('n'):
        print('Pushing with tags')
        subprocess.run(['git', 'push'], check=True)
        subprocess.run(['git', 'push', '--tags'], check=True)

    release_response = input('Would you like to update the release branch? [Y/n]:')
    if not release_response.lower().startswith('n'):
        current_branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                        check=True, capture_output=True)
        current_branch = current_branch.stdout
        subprocess.run(['git', 'checkout', 'release'], check=True)
        subprocess.run(['git', 'push'], check=True)
        subprocess.run(['git', 'checkout', current_branch], check=True)

    if do_stash:
        print('Unstashing changes')
        subprocess.run(['git', 'stash', 'pop'])


if __name__ == '__main__':
    main()
