#!/usr/bin/env python
from .backend import BackendBase

class Shell(BackendBase):
    def __init__(self, session_config, **args):
        super().__init__(session_config=session_config)
    
    def run(self):
        while True:
            query = input(":: ")
            if query in ["reset"]:
                self.init_session()
                print("reseted all")
                continue
            response = self.session(query)
            print(response)

