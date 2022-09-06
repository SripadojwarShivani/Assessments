#Author: Shivani Sripadojwar



#Including FastAPI 

from fastapi import FastAPI

app=FastAPI()
list_of_users=list()

#get method

@app.get("/home/{username}")
def get_data(username:str):
	return{
	"Successfully fetched details of user: "+username
	}


#post method

@app.post("/post")
def post_data(username:str):
	list_of_users.append(username)
	return{
			"usernames":list_of_users
		  }


#put method

@app.put("/username/{username}")
def put_data(username:str):
	print(username)
	list_of_users.append(username)
	return{"username":username}
