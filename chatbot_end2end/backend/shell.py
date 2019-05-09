#!/usr/bin/env python

class Shell:
    def __init__(self, interact_session):
        self.session = interact_session
    
    def run(self):
        while True:
            query = input(":: ")
            response = self.session.response(query)
            print(response)
    
