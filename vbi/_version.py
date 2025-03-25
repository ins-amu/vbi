__version__ =  "v0.2"

if __name__ == '__main__':
    print(__version__)
    import os, sys
    _, cmd, *args = sys.argv
    print(cmd, args)
    if cmd == 'tag':
        import subprocess
        def do(cmd):
            print(cmd)
            subprocess.check_call(cmd.split(' '))
        do(f'/usr/bin/git add netsci/_version.py')
        do(f'/usr/bin/git commit -m bump-version')
        do(f'/usr/bin/git push -u origin main')
        do(f'/usr/bin/git tag -f {__version__}')
        do(f'/usr/bin/git push -u origin -f {__version__}')