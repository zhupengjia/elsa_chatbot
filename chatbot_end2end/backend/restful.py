#!/usr/bin/env python

from flask import Flask, request, jsonify, Response

class Restful:
    def __init__(self, interact_session, cfg):
        self.session = interact_session
        self.cfg = cfg
        self.app = Flask(__name__)
    
    @self.app.route('/robot/api/query', methods=['POST'])
    def get_response(self):
        query = request.form.get('text').strip()
        session_id = request.form.get('sessionId', "123456")
        response = self.session.response(query, session_id)
        if response is None:
            return Response(json.dumps({"code":0, "message":"200 OK", 'sessionId':session_id, "data":{"response": ":)"}}), mimetype='application/json')
        return Response(json.dumps({"code":0, "message":"200 OK", 'sessionId':session_id, "data":{"response": response}}), mimetype='application/json')

    def run(self):
        self.app.run(host='0.0.0.0', port=self.cfg.port, threaded=True)


