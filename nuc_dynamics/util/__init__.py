import sys


def warn(msg, prefix='WARNING'):
    print('%8s : %s' % (prefix, msg))


def critical(msg, prefix='ABORT'):
    print('%8s : %s' % (prefix, msg))
    sys.exit(0)


def test_imports(gui=False):
    import sys
    from distutils.core import run_setup
    
    critical = False
    
    try:
        import numpy
    except ImportError as err:
        critical = True
        warn('Critical Python module "numpy" is not installed or accessible')

    try:
        import cython
    except ImportError as err:
        critical = True
        warn('Critical Python module "cython" is not installed or accessible')
    
    try:
        from .. import dyn_util
    except ImportError as err:
        import os
        cwd = os.getcwd()
        try:
            fdir = os.path.dirname(os.path.normpath(__file__))
            print(fdir)
            os.chdir(fdir)
            warn('Utility C/Cython code not compiled. Attempting to compile now...')        
            run_setup('setup_cython.py', ['build_ext', '--inplace'])
        finally:
            os.chdir(cwd)
        
        try:
            import dyn_util
            warn('NucDynamics C/Cython code compiled. Please re-run command.')
            sys.exit(0)
            
        except ImportError as err:
            critical = True
            warn('Utility C/Cython code compilation/import failed')     
        
    if critical:
        warn('NucDynamics cannot proceed because critical Python modules are not available', 'ABORT')
        sys.exit(0)
