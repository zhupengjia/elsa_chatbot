#!/usr/bin/env python3
from flask import Flask, request, Response, send_from_directory
import json

app = Flask(__name__)

@app.route('/api/query', methods=['POST', 'GET'])
def query():
    req_args = request.args
    query = req_args.get('query')
    session_id = req_args.get('session')
    print(query, session_id)
    return Response(json.dumps({"response": "return: "+query, "session":session_id}), mimetype='application/json') 

if __name__ == '__main__':
    print('>>>>>>>>>>Flask server is running on 0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)


