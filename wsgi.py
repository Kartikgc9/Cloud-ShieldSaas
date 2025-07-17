import os
from app import app
from config import config

# Set configuration based on environment
config_name = os.getenv('FLASK_CONFIG') or 'default'
app.config.from_object(config[config_name])

if __name__ == "__main__":
    app.run()