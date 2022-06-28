#             Assignment 1
#             A1:Spice - Part 1
#             Name: C. Dheeraj Sai
#             Roll No.: EE20B029


from sys import argv, exit


CIRCUIT = ".circuit"
END = ".end"


# Checking if there are only 2 command line arguements and the correct netlist file type to be accessed by the interpretter

if len(argv) != 2:
    print(f"\nUsage: python {argv[0]} <netlist file name>\n")
    exit(0)
if not argv[1].endswith(".netlist"):  # Checking file type
    print("\nWrong file type\n")
    exit(0)

####################################################################
####################################################################

try:
    with open(argv[1]) as f:
        cktlines = f.readlines()
        start = -1
        end = -2

        # Extracting index of the starting and ending valid circuit lines from the netlist file
        for line in cktlines:
            # if CIRCUIT == line.split("#")[0].split("\n")[0]:
            if CIRCUIT == line.split("#")[0].split()[0]:
                start = cktlines.index(line)
            elif END == line.split("#")[0].split()[0]:
                end = cktlines.index(line)
        # Checking if the circuit specifications are correct
        if start >= end:
            print("\nInvalid circuit block\n")
            exit(0)
        else:
            # declaring variables to be used later
            tokens = []
            controller_nodes = []
            voltage_label = ""
            # removing the comments and whitespaces at the ends from each line of the netlist file
            for line in cktlines[start + 1 : end]:
                line = line.split("#")[0]
                line = line.strip()
                # Ignoring empty lines
                if line == "":
                    continue
                # Checking if the component has valid no. of parameters and storing them in their respective variables
                component_name = line.split()[0]
                if component_name[0] in ("R", "L", "C", "V", "I"):
                    if len(line.split()) == 4:
                        component_nodes = line.split()[1:3]
                    else:
                        print(f"\nIncorrect number of paramters for {component_name}\n")
                        exit(0)
                elif component_name[0] in ("H", "F"):
                    if len(line.split()) == 5:
                        component_nodes = line.split()[1:3]
                        if line.split()[3][0] == "V":
                            voltage_label = line.split()[3]
                        else:
                            print("\nInvalid voltage designation\n")
                            exit(0)
                    else:
                        print(f"\nIncorrect number of paramters for {component_name}\n")
                        exit(0)
                elif component_name[0] in ("E", "G"):
                    if len(line.split()) == 6:
                        component_nodes = line.split()[1:3]
                        controller_nodes = line.split()[3:5]
                    else:
                        print(f"\nIncorrect number of paramters for {component_name}\n")
                        exit(0)
                else:
                    print(f"{component_name} is not a valid representation")
                    exit(0)
                # Checking if the nodes provided are alphanumeric
                for nodes in component_nodes + controller_nodes:
                    if nodes.isalnum() == False:
                        print(
                            "\nInvalid node designation. Nodes should be named by alphanumeric charectors\n"
                        )
                        exit(0)
                component_value = line.split()[-1]
                ####################################################################
                # Extracting tokens from these lines and storing them as a list
                token = [
                    component_name,
                    component_nodes,
                    controller_nodes,
                    voltage_label,
                    component_value,
                ]
                tokens.append(token)
            ####################################################################
            # printing the tokens as per the required format
            print("")
            for component in reversed(tokens):
                for elem in reversed(component):
                    if isinstance(elem, list):
                        elem = " ".join(elem[::-1])
                    if elem in ("", []):
                        continue
                    print(f"{elem} ", end="")
                print("")
            print("")

# Printing error in case of iinvalid file name or location
except IOError:
    print("\nInvalid file name/location\n")
    exit(0)
