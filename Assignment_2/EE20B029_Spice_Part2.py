#             Assignment 2
#             A2:Spice - Part 2
#             Name: C. Dheeraj Sai
#             Roll No.: EE20B029


import numpy as np
from sys import argv, exit

CIRCUIT = ".circuit"
END = ".end"
AC = ".ac"

# Checking no. of commandline arguments
if len(argv) != 2:
    print(f"\nUsage: python {argv[0]} <netlist file name>\n")
    exit(0)

# Checking file type
if not argv[1].endswith(".netlist"):
    print("\nWrong file type\n")
    exit(0)

# Class for all the passive components
class PassiveComponent:
    def __init__(self, tempList):
        (
            self.name,
            self.from_node,
            self.to_node,
            self.val,
        ) = tempList
        self.val = float(self.val)


# Class for voltage sources and current sources
class ActiveComponent:
    def __init__(self, tempList):
        if ac:
            (
                self.name,
                self.from_node,
                self.to_node,
                self.type,
                self.val,
                self.phase,
            ) = tempList
            self.phase = float(self.phase)
            self.val = (float(self.val) / 2) * complex(
                np.cos(self.phase), np.sin(self.phase)
            )
        else:
            (
                self.name,
                self.from_node,
                self.to_node,
                self.type,
                self.val,
            ) = tempList
            self.val = float(self.val)


# Class for controlled sources
class ControlledComponent:
    def __init__(self, tempList):
        (
            self.name,
            self.from_node,
            self.to_node,
            self.n1,
            self.n2,
            self.val,
        ) = tempList
        self.val = float(self.val)


### Functions for filling the coeffiecient matrix with appropriate submatrices ###

# Function for adding submatrix stamps of passive elements
def addStampPassive(factorMatrix, from_node, to_node, val):
    factorMatrix[from_node, from_node] += 1 / val
    factorMatrix[from_node, to_node] += 1 / val * (-1)
    factorMatrix[to_node, from_node] += 1 / val * (-1)
    factorMatrix[to_node, to_node] += 1 / val

    return factorMatrix


# Function for adding submatrix stamps for current sources
def addCurrentSource(indMatrix, from_node, to_node, val):
    indMatrix[from_node, 0] += val
    indMatrix[to_node, 0] += val * (-1)
    return indMatrix


# Function for adding submatrix stamps for voltage sources
def addVoltageSource(factorMatrix, indMatrix, from_node, to_node, val, n, r):
    factorMatrix[to_node, n + r] += -1
    factorMatrix[n + r, to_node] += -1
    factorMatrix[from_node, n + r] += 1
    factorMatrix[n + r, from_node] += 1
    indMatrix[n + r, 0] += val
    return [factorMatrix, indMatrix]


# Function for addingn submatrix stamps of VCVS
def addVCVS(factorMatrix, from_node, to_node, n1, n2, val, n, r):
    factorMatrix[n + r, from_node] += val * (-1)
    factorMatrix[n + r, to_node] += val
    factorMatrix[n + r, n1] += 1
    factorMatrix[n + r, n2] += -1
    factorMatrix[n1, n + r] += 1
    factorMatrix[n2, n + r] += -1
    return factorMatrix


# Function for adding submatrix stamps of VCCS
def addVCCS(factorMatrix, from_node, to_node, n1, n2, val):
    factorMatrix[n1, from_node] += val
    factorMatrix[n2, from_node] += val * (-1)
    factorMatrix[n1, to_node] += val * (-1)
    factorMatrix[n2, to_node] += val
    return factorMatrix


# Function for adding submatrix stamps of CCVS
def addCCVS(factorMatrix, from_node, to_node, n1, n2, val, n, r):
    factorMatrix[n + r, from_node] += 1
    factorMatrix[n + r, to_node] += -1
    factorMatrix[n + r + 1, n1] += 1
    factorMatrix[n + r + 1, n2] += -1
    factorMatrix[from_node, n + r] += 1
    factorMatrix[to_node, n + r] += -1
    factorMatrix[n1, n + r + 1] += 1
    factorMatrix[n2, n + r + 1] += -1
    factorMatrix[n + r + 1, n + r] += val
    return factorMatrix


# Function for adding submatrix stamps of CCCS
def addCCCS(factorMatrix, from_node, to_node, n1, n2, val, n, r):
    factorMatrix[n + r, from_node] += 1
    factorMatrix[n + r, to_node] += -1
    factorMatrix[from_node, n + r] += 1
    factorMatrix[to_node, n + r] += -1
    factorMatrix[n1, n + r] += val
    factorMatrix[n2, n + r] += -val
    return factorMatrix


try:
    with open(argv[1]) as f:
        cktlines = f.readlines()
        start = -1  #               variable for storing start index of netlist lines
        end = -2  #                 variable for storing end index of netlist lines
        componentList = []  # variable for storing all objects of components in a list
        n = 0  #                    variable for storing no. of KCL equations
        k = 0  #                    variable for storing no. of voltage sources
        ac = False  #               variable for checking if independent sources are AC
        nodes = {}  # dictionary containing all node names corresponding to node values
        for line in cktlines:
            tempList = line.split("#")[0].split()
            if CIRCUIT == tempList[0]:
                start = cktlines.index(line)
            elif END == tempList[0]:
                end = cktlines.index(line)
            elif AC == tempList[0]:
                ac = True
                freq = 2 * np.pi * float(tempList[2])
        if start >= end:
            print("\nInvalid circuit block\n")
            exit(0)

        # extracting all the circuit components into their respective objects
        for line in cktlines[start + 1 : end]:
            tempList = line.split("#")[0].split()
            if tempList[0][0] in ("R", "L", "C"):
                component = PassiveComponent(tempList)
            elif tempList[0][0] in ("V", "I"):
                if tempList[0][0] == "V":
                    k = k + 1
                component = ActiveComponent(tempList)
            elif tempList[0][0] in ("E", "G"):
                if tempList[0][0] == "E":
                    k = k + 1
                component = ControlledComponent(tempList)
            elif tempList[0][0] in ("H", "F"):
                if tempList[0][0] == "H":
                    k = k + 1
                for line2 in cktlines[start + 1 : end]:
                    tempList2 = line2.split("#")[0].split()
                    if tempList[3] == tempList2[0]:
                        n1 = tempList2[1]
                        n2 = tempList2[2]
                        if n1 == "GND":
                            n1 = 0
                        if n2 == "GND":
                            n2 = 0
                        n1 = int(n1)
                        n2 = int(n2)
                    else:
                        continue
                    del tempList[3]
                    tempList.insert(3, n1)
                    tempList.insert(4, n2)
                component = ControlledComponent(tempList)
                k = k + 1
            # variable k stores total no. of voltage sources and it is also contributed by VCVS
            componentList.append(component)

            # checks if nodes are alphanumeric
            if not component.from_node.isalnum() and component.to_node.isalnum():
                print(
                    "\nInvalid node designation. Nodes should be named by alphanumeric charectors\n"
                )
                exit(0)

            # extracting nodes into a dictionary
            if component.from_node not in nodes:
                if component.from_node == "GND":
                    nodes["n0"] = 0
                else:
                    nodes[f"n{component.from_node}"] = int(component.from_node)
            if component.to_node not in nodes:
                if component.to_node == "GND":
                    nodes["n0"] = 0
                else:
                    nodes[f"n{component.to_node}"] = int(component.to_node)

        # constructing two 0x0 matrics using the dimensions determined from n and k
        n = len(nodes)
        M = np.zeros(((n + k), (n + k)), dtype=complex)
        b = np.zeros(((n + k), 1), dtype=complex)
        vlist = []  # variable for storing voltage sources
        r = 0

        for component in componentList:
            if component.from_node == "GND":
                component.from_node = 0
            elif component.to_node == "GND":
                component.to_node = 0
            component.from_node = int(component.from_node)
            component.to_node = int(component.to_node)

            ###Extracting the component stamps into the coefficient matrix using the functions defined above###
            if component.name[0] in ("R", "L", "C"):
                if component.name[0] == "L" and ac:
                    Xl = component.val * freq
                    component.val = complex(0, Xl)
                elif component.name[0] == "C" and ac:
                    Xc = 1 / (component.val * freq) * (-1)
                    component.val = complex(0, Xc)
                M = addStampPassive(
                    M, component.from_node, component.to_node, component.val
                )
            elif component.name[0] == "I":
                b = addCurrentSource(
                    b, component.from_node, component.to_node, component.val
                )
            elif component.name[0] == "V":
                M, b = addVoltageSource(
                    M, b, component.from_node, component.to_node, component.val, n, r
                )
                r = r + 1
                vlist.append(component.name)
            elif component.name[0] == "E":
                M = addVCVS(
                    M,
                    component.from_node,
                    component.to_node,
                    component.n1,
                    component.n2,
                    component.val,
                    n,
                    r,
                )
                r = r + 1
            elif component.name[0] == "G":
                M = addVCCS(
                    M,
                    component.from_node,
                    component.to_node,
                    component.n1,
                    component.n2,
                    component.val,
                    n,
                    r,
                )
            elif component.name[0] == "H":
                M = addCCVS(
                    M,
                    component.from_node,
                    component.to_node,
                    component.n1,
                    component.n2,
                    component.val,
                    n,
                    r,
                )
                r = r + 2
            elif component.name == "F":
                M = addCCCS(
                    M,
                    component.from_node,
                    component.to_node,
                    component.n1,
                    component.n2,
                    component.val,
                    n,
                    r,
                )
                r = r + 1
        # Taking ground voltage to be 0
        M[0] = 0
        M[0, 0] = 1
        # solving the matrix equation and printing the the solution matrix
        solution = np.linalg.solve(M, b)
        print(f"\nSolution matrix:\n\n{solution}")

        # printing all the unknown node voltages and current through voltage sources
        print("\nNode voltages:\n")
        for i in range(n):
            print(f"Vn{i} = {solution[i][0]}")
        print(f"\nCurrent through voltage sources:\n")
        for j in range(k):
            print(f"I[{vlist[j]}] = {solution[n+j][0]}")
        print("")

# Exits program if the file name or the location is incorrect
except IOError:
    print("\nInvalid file name/location\n")
    exit(0)
