# Enter your code here. Read input from STDIN. Print output to STDOUT

import fileinput

for line in fileinput.input():
    time = float(line)
    if time > 4:
        print('8.00')
    else:
        print(2 * time)