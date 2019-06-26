#!/usr/bin/env python

class Shell:
    def __init__(self, interact_session, **args):
        self.session = interact_session
    
    def run(self):
        while True:
            query = input(":: ")
            response = self.session.response(query)
            print(response)

    def query(self, text):
        response = self.session.response(text)
        print(response)
        

