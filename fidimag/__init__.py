import os
import sys
import platform
try:
    from . import extensions
    from . import common
    from . import atomistic
    from . import micro
except ImportError as e:
    # cwd = os.getcwd()
    FIDIMAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # os.chdir(FIDIMAG_DIR)
    message = """
    Could not load the Fidimag extensions. This could be due to a few things:
      1) The extensions are not built. Try running 'make' from the root directory.

      2) FIDIMAG loads comon libraries dynamically. Currently, FIDIMAG requires
         * Sundials 2.6.2
         * FFTW 3.3.5
         If you are using Linux, you may be able to download these through a
         package manager such as apt or yum. However, convenience scripts
         are provided in the folder bin to install these locally for Ubuntu.

      3) If you've run the scripts to install the dynamic libaries,
         you may still need to export the LD_LIBRARY_PATH environment
         variable. You can do this from the shell with:
         export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH

      4) If after trying *all* of the above, you still can't get Fidimag to install,
         please create an issue on the following website:

         https://github.com/computationalmodelling/fidimag/issues

         Copy and paste *all* of the information below which will help 
         us to diagnose your problem:


         Exception Error message 
         -----------------------
         {}

         System Info
         -----------------------
         Architecture: {}
         Platform: {}
         Processor: {}
         """
    print(message.format(FIDIMAG_DIR, e, platform.machine(), platform.platform(), platform.processor()))

    # os.system("make build")
    #print("Building extensions done.")
    #os.chdir(cwd)
# try:
#    from . import extensions
#    from . import common
#    from . import atomistic
#    from . import micro
#except ImportError as e:
#    raise ImportError("Loading the FIDIMAG extensions still didn't work."
#                      "\nTry the steps above!")


from .gui import GUI

citation = common.citation

__version__ = '0.2'
