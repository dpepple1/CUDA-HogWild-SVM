'''
A program to read in sparse data (RCV1) in the form:
(INDEX):(VALUE) and label the rows with the number
of non-zero items

By Derek Pepple
'''
import sys


def main():
    if len(sys.argv) < 3:
        print("Error: missing arguments!")

    else:
        r_file_path = sys.argv[1]
        w_file_path = sys.argv[2]
        with open(r_file_path, 'r') as rfh:
            with open(w_file_path, 'w') as wfh:
                line = rfh.readline()
                while line:
                    pattern = line.split(" ")
                    components = len(pattern) - 1
                    #print(components, line)
                    wfh.write(str(components) + " " + line)
                    line = rfh.readline()
                    

if __name__ == "__main__":
    main()
