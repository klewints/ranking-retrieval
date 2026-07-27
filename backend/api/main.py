from fastapi import FastAPI
from backend.api.routers.search_router import router as search_router

app = FastAPI(title='Music Recommendation Search', version='0.1')


@app.get('/')
def root():
    return {
        'status': 'running',
        'service': 'music search API'
    }


app.include_router(search_router)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.api.main:app', host='127.0.0.1', port=8000, reload=False)
