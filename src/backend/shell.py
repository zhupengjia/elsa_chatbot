#!/usr/bin/env python

class Shell:
    def __init__(self):
        pass
    
    def reset(self):
        return "reset"

    def interact(self, query):
        return "response"

    def run(self):
        while True:
            query = input(":: ")
            query = query.strip()
            if len(query) < 1:
                continue

            response = self.interact(query)
            print(response)

