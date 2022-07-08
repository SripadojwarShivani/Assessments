#Author: Shivani Sripadojwar
#email:ssripadojwar@vsoftconsulting.com

#Writing different functions in calculas module

#fibonacci 

def fibonacci(p):
	if p<=2:
		return 1
	else:
		return fibonacci(p-1)+fibonacci(p-2)

#armstrong number

def armstrong(n):
	temp=n
	l=len(str(n))
	sum=0
	while n>0:
		rem=n%10
		sum=sum+(rem)**l
		n=n//10
	if(temp==sum):
		print("armstrong number!")
	else:
		print("Not an armstrong!")



#factorial

def factorial(n):
	fact=1
	for i in range(1,n+1):
		fact=fact*i
	print("factorial of the given number: "+str(fact))



#palindrome

def palindrome(s):
	m=s
	word=""
	for i in s:
		word=i+word
	if m==word:
		print("it is palindrome")
	else:
		print("it is not palindrome")
