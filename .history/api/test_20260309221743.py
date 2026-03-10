"""Minimal Vercel Python test - no dependencies."""

import json

def handler(event, context):
    """Simple Vercel handler returning JSON."""
    # Parse the request path from event
    path = event.get('path', '/')
    method = event.get('httpMethod', 'GET')
    
    if path == '/api/test' and method == 'GET':
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "ok", "message": "Python handler works!"})
        }
    else:
        return {
            "statusCode": 404,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"detail": "Not Found"})
        }

