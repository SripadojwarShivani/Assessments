#Author: Shivani Sripadojwar



#including different modules

import csv
import json
import math
import random
 
# including own module
import calculas

#Reading and Writing of different file formats

#Reading and Writing text files

message="Here we are writing something into the text file."

#Writing text file
file1=open("sample.txt","w")
file1.write(message)
file1.close()

#Reading text file
file1=open("sample.txt","r")
print((file1.read()))
file1.close()


#Reading and Writing CSV files

labels=['Ename','Eid','Edept','Esal']

data=[['Ted','0001','operations','500000'],
      ['Cynthia','0002','sales','700000'],
      ['David','0003','marketing','590000'],
      ['Amrrah','0004','Hr','80000']]


# Writing CSV file 
with open('sample.csv','w') as file1:
	newfile=csv.writer(file1)
	newfile.writerow(labels)
	newfile.writerows(data)


#Reading CSV file
with open ('sample.csv','r') as file1:
	newfile=csv.reader(file1)
	for lines in newfile:
		print(lines)


#Reading and Writing JSON files

# Data to be written
dictionary={"name":"sam","rollno" :56,"cgpa" :9.8, "phonenumber" :"9976770500"}  

json_object= json.dumps(dictionary, indent= 4)  

#Writing JSON file
with open("sample.json","w") as outfile:    
	outfile.write(json_object)

#Reading JSON file
with open("sample.json",'r') as outfile:
	json_object=json.load(outfile)
print(json_object)

#Exceptional Handling

def fun(a):
    if a <4:         
    # throws ZeroDivisionError for a = 3        
    	b= a/(a-3)     
    # throws NameError if a >= 4    
    print(("Value of b = ", b))     
try:    
    fun(3)    
    fun(5) 
    # note that braces () are necessary here for
    # multiple exceptions
except ZeroDivisionError:    
    print("ZeroDivisionError Occurred and Handled")
except NameError:    
    print("NameError Occurred and Handled")
finally:
	print("Error Handled")


#Calling different functions from calculas module

print((calculas.fibonacci(9)))
calculas.armstrong(1634)
calculas.factorial(5)
calculas.palindrome("liril")

#Calling different functions from built-in-modules(Math module,Random module)
print((math.pow(2,3)))
print((math.floor(5.6)))
print((math.sqrt(9)))
r=random.randint(2,9)
print(r)
y=random.choices(['a','b','c','d','e','f','g','h'])
print(y)
