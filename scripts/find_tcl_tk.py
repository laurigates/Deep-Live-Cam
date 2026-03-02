"""Print Tcl and Tk library paths for the running Python installation."""
import os
import sys
import glob

def find_tcl():
    hits = (
        glob.glob(os.path.join(sys.base_prefix, "lib", "tcl*", "init.tcl"))
        + glob.glob(os.path.join(sys.base_prefix, "tcl", "tcl*", "init.tcl"))
    )
    return os.path.dirname(hits[0]) if hits else ""

def find_tk():
    hits = (
        glob.glob(os.path.join(sys.base_prefix, "lib", "tk[0-9]*"))
        + glob.glob(os.path.join(sys.base_prefix, "tcl", "tk[0-9]*"))
    )
    return hits[0] if hits else ""

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "tcl"
    print(find_tcl() if which == "tcl" else find_tk())
