# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:47:52 2020

@author: wyckliffe
"""


import uvicorn
from fastapi import FastAPI, Path, Request
from fastapi.responses import JSONResponse

import runner
from models import MyException, Configuration, Index

app = FastAPI(title='SMART Data Science Application',
              description='A Smart Data Science Application running on FastAPI + uvicorn',
              version='0.0.1')



@app.get("/{index}")
async def get_result(index: Index = Path(..., title="The name of the Index")
                     ):
                config = Configuration(
                    index=index
                        )
                try:
                    result = await runner.run(config)
                    return JSONResponse(status_code=200, content=result)
                except Exception as e:
                    raise MyException(e)

@app.exception_handler(MyException)
async def unicorn_exception_handler(request: Request, exc: MyException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Error occurred! Please contact the system admin."},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)