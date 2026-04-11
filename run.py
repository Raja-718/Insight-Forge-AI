"""
run.py — Grivora AI entry point
Registers both the main blueprint (app/routes.py) and the new auth blueprint.
"""
from app import create_app
from auth_system.auth_routes import auth_bp

app = create_app()

# Register auth blueprint — NEW, non-invasive
app.register_blueprint(auth_bp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
