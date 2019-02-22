#!/usr/bin/env python3
from flask import Flask, request, Response, send_from_directory
import json, random

app = Flask(__name__)

@app.route('/api/query', methods=['POST'])
def query():
    query = request.form.get('text')
    session_id = request.form.get('session')
    if session_id is None:
        session_id = str(random.randint(a=0, b=100000000))
    return Response(json.dumps({"code":0, "message":"200 OK", 'session':session_id, "data":{"response": "return: "+query}}), mimetype='application/json') 

if __name__ == '__main__':
    print('>>>>>>>>>>Flask server is running on 0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)


