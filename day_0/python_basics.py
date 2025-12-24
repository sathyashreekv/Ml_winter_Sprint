#printing 
print("hello world")

import sys
print(sys.version)

#variables
#variable_name=value
x=5
x=1.5
print(x)
name="sathya"
print(name)
#printing onsame line
print("sathya",end="")
print("shree")

#printing numbers
print(30)
print((9-0))

#mixing number and text
print("I am sathya 19 years old")
print("i am ",19,"years old")

#casting 
age=str(13)
num=int(age)
a=float(age)
print(age,num,a)

#getting type
print(type(a))

#multivalue assignment
s,t,u=1,3.4,"o"
print(s,t,u)

#1- many 
x=y=z=1111
print(x,y,z)

#impppp unpacking a collection

fruit=["apple","grapes","kiwi"]
first,second,third=fruit
print(first,second,third)

print(type(fruit))
print("i am "+fruit[0]+" i am eating "+second)


x=memoryview(b'hello')
print(x)

print(type(x))

print(id(x))

print(complex(3,5))
n=b'sathya'

print(n)
print(bin(5))
print(bin(4))
print(bytearray(5))
print(bytearray(b'sathya'))

#random module

import random
print(random.randint(1,256))


#-----strings----
str1="sathya"
str2="shree"
str3="""
Rvu library is very good place 
to study
"""
print(str3)

print("i am unstoppable, i am sathya")

#py as no datatype named char so uwe need to use indexing

char_first=str3[1]
print(char_first)

#looping through a string
for char in str3:
    print(char)

#sTRINg length

a=len(str3)
print(a)

st4="abc "
print(len(st4))

#to check string is present in inside

txt="the best things in life are free"
print("free" in txt)

if "free" in txt:
    print("yes it is ")

if "expensive" not in txt:
    print("no it is not")
else:
    print("yes")
#not in

#slicing
b=txt[::-1]
print(b)

c=txt[1:5]
#1 included but not 5
print(c)

print(c.upper())
print(c.lower())


#removing whitespace before nad after
g="  hello world  "
print(g.strip())

#replace(old,new)
print(g.replace("world","sathya"))

#split 
print(g.split(" "))

fr=f"My  name is {third} and i am {s} years old"
print(fr)


n1=8
n2=9

print(n1+n2)
print(n1-n2)
print(n1*n2)
print(n1/n2)
print(n1%n2)
print(n1**n2)
print(n1//n2)

print(n1>0 and n1<n2)
print(n2>n1 or n1>0)

mylist=["apple","banana","cherry"]
print(mylist)


list1=[1,2,3,4,5]
print(list1)

mixedlist=[1,"sathya",3.4,True]
print(mixedlist)

print(type(mixedlist))
print(len(mixedlist))

thislist=list(("apple","banana","cherry"))
print(thislist)

print(thislist[0])
print(thislist[-1])
print(thislist[0:2])

if "apple"in thislist:
    print("yes , apple in the fruits list")
else:
    print("no apple is not in the list")

thislist[1]='blackcurrent'
print(thislist)

thislist[1:2]=['blackcurrent','mango']
print(thislist)

thislist[1:3]=['kiwi']
print(thislist)
thislist.insert(2,'watermelon')
print(thislist)

thislist.append('orange')
print(thislist)

tropical=['grapes','pomogranate']
thislist.extend(tropical)
print(thislist)


thislist.remove('cherry')#removes a specified item and for duplicate values it removes the first occurence
print(thislist)

thislist.pop(1)#pop() removes the specified index
print(thislist)

thislist.pop() #pop() removes last item if not specified
print(thislist)

del thislist[0] #del keyword used to delete specific element or entire list
print(thislist)

thislist.clear()#this clear the list content but still list exist
print(thislist)

for fruit in tropical:
    print(fruit)  
for i in range(len(tropical)):
    print(tropical[i])
[print(x)for x in tropical]

fruits=["apple","banana","cherry","kiwi","mango"]
newlist=[x for x in fruits if "a" in x]
print(newlist)
fruits.sort() #ascending order
print(fruits)
newlist1=[x**2 for x in range(10) if x%2==0]
print(newlist1)
newlist.sort() #ascending order
print(newlist1)

#sort(reverse=True) #descending order
#sort(my function any cutomised function)
newlist1.reverse() #reverse order
print(newlist1)

thistuple=("apple","banana","cherry")
print(thistuple)
print(len(thistuple))
print(type(thistuple))

print(thistuple[1])

current_lit=list(thistuple)
current_lit[2]='orange'
thistuple=tuple(current_lit)
print(thistuple)





