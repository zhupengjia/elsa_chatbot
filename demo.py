#!/usr/bin/env python
import sys, os


class InteractiveSession():
    def __init__(self):
        pass

    def interact(self):
        #interaction loop
        while True:
            # get input from user
            u = input(':: ')
            u = u.strip()
            if len(u) < 1:
                continue
            
            print(u)



if __name__ == '__main__':
    # create interactive session
    isess = InteractiveSession()
    # begin interaction
    isess.interact()

     

