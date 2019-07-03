#!/usr/bin/env python
from .backend import BackendBase
from flask import Flask, request, json, Response

class Restful(BackendBase):
    def __init__(self, session_config, url_rule, methods, port=5000, **args):
        super().__init__(session_config=session_config, **args)
        self.url_rule = url_rule
        self.methods = methods
        self.port = port
        self.app = Flask(__name__)
        self.app.add_url_rule(self.url_rule, methods=self.methods, view_func=self.get_response)

    def get_response(self):
        query = request.form.get('text').strip()
        session_id = request.form.get('sessionId', "123456")
        if query in ["reset"]:
            self.init_session()
            return Response(json.dumps({"code":0, "message":"200 OK", 'sessionId':session_id, "data":{"response": "reseted all"}}), mimetype='application/json')

        response = self.session(query, session_id=session_id)
        if response is None:
            return Response(json.dumps({"code":0, "message":"200 OK", 'sessionId':session_id, "data":{"response": ":)"}}), mimetype='application/json')
        return Response(json.dumps({"code":0, "message":"200 OK", 'sessionId':session_id, "data":{"response": response}}), mimetype='application/json')

    def run(self):
        self.app.run(host='0.0.0.0', port=self.port, threaded=True)


