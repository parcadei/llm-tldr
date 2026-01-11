#!/usr/bin/env python3
"""
TLDR Brain Server - Flask API for visualization
"""

from flask import Flask, jsonify, request, send_file, send_from_directory
from pathlib import Path
import json
import subprocess
import os


def create_app(project_root: Path = None, brain_file: Path = None):
    """Factory function to create Flask app with configurable paths.
    
    Args:
        project_root: Project directory (default: current directory)
        brain_file: Path to brain.json (default: project_root/brain.json)
    
    Returns:
        Configured Flask app
    """
    if project_root is None:
        project_root = Path.cwd()
    if brain_file is None:
        brain_file = project_root / "brain.json"
    
    # Find brain_ui directory (in same folder as this script)
    script_dir = Path(__file__).parent
    static_folder = script_dir / "brain_ui"
    
    app = Flask(__name__, static_folder=str(static_folder), static_url_path="")
    
    # Store paths in app config
    app.config['PROJECT_ROOT'] = project_root
    app.config['BRAIN_FILE'] = brain_file

    @app.route("/")
    def index():
        return send_from_directory(str(static_folder), "index.html")

    @app.route("/api/brain")
    def get_brain():
        bf = app.config['BRAIN_FILE']
        if not bf.exists():
            return jsonify({"error": "brain.json not found. Run 'tldr brain build' first."}), 404
        with open(bf, "r") as f:
            return jsonify(json.load(f))

    @app.route("/api/source/<path:filepath>")
    def get_source(filepath):
        pr = app.config['PROJECT_ROOT']
        # Security: Ensure path is within project
        target = (pr / filepath).resolve()
        if not str(target).startswith(str(pr)):
            return jsonify({"error": "Access denied"}), 403
        
        if not target.exists():
            return jsonify({"error": "File not found"}), 404
            
        return send_file(target, mimetype="text/plain")

    @app.route("/api/search")
    def semantic_search():
        query = request.args.get("q")
        if not query:
            return jsonify({"error": "Missing query parameter 'q'"}), 400
        
        pr = app.config['PROJECT_ROOT']
        cmd = f'tldr semantic search "{query}" --path "{pr}"'
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(pr))
            return jsonify({"results": result.stdout.splitlines()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/impact/<func_name>")
    def impact(func_name):
        pr = app.config['PROJECT_ROOT']
        cmd = f"tldr impact {func_name} ."
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(pr))
            return jsonify({"output": result.stdout})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# For backwards compatibility and direct script execution
app = None
PROJECT_ROOT = Path.cwd()
BRAIN_FILE = PROJECT_ROOT / "brain.json"


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
