"""Local Uvicorn entry point; cloud startup uses the factory explicitly."""
from telecom.api_factory import create_app

app = create_app()
