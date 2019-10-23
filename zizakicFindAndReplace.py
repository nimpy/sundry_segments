import fileinput
import sys

assert len(sys.argv) == 2, "The first argument of this script should be the name of the (.bib) file on which you want to perform the Žižakić -> Zizakic find and replace."

with fileinput.FileInput(sys.argv[1], inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace("Žižakić", "Zizakic"), end='')

