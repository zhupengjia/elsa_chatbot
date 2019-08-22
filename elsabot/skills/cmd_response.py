#!/usr/bin/env python
import re, traceback
from .skill_base import SkillBase
"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
    Response skill for Commands
"""

class CMDResponse(SkillBase):
    '''
        Special commands for current_status
    '''
    def __init__(self, skill_name, **args):
        super().__init__(skill_name)

    def update_response(self, response, current_status):
        current_status["$REDIRECT_SESSION"] = None
        utterance = current_status["$UTTERANCE"].strip()
        cmd = [s.strip() for s in re.split("\s", utterance) if s.strip()]
        if cmd[0] in ["`clear", "`restart", "`exit", "`stop", "`quit"]:
            current_status["$SESSION_RESET"] = True
        elif cmd[0] in ["`redirect", "`connect"]:
            if len(cmd) > 1:
                current_status["$REDIRECT_SESSION"] = cmd[1]
                current_status["$RESPONSE"] = "Connect to {}".format(cmd[1])
            else:
                current_status["$REDIRECT_SESSION"] = None
        elif cmd[0] in ["`list"]:
            current_status["$RESPONSE"] = str("Available variables:\n {}".format("\n".join(list(current_status.keys()))))
        elif cmd[0] in ["`get"]:
            if len(cmd) < 2:
                current_status["$RESPONSE"] = str(current_status)
            elif cmd[1] in current_status:
                current_status["$RESPONSE"] = str(current_status[cmd[1]])
            else:
                current_status["$RESPONSE"] = str("Available variables:\n {}".format("\n".join(list(current_status.keys()))))
        elif cmd[0] in ["`set"] and len(cmd)>2:
            try:
                current_status[cmd[1]] = eval(cmd[2:])
                current_status["$RESPONSE"] = current_status[cmd[2:]]
            except:
                current_status["$RESPONSE"] = traceback.format_exc()
        elif cmd[0] in ["`history"]:
            history = "\n".join(["======{}\n- {}\n * {}".format(x[0], x[1], x[2]) for x in current_status["$HISTORY"]])
            current_status["$RESPONSE"] = history
        elif cmd[0] in ["`help"]:
            current_status["$RESPONSE"] = """
                Commands:
                    `clear, `restart, `exit, `stop, `quit, `q: reset the session
                    `redirect, `connect USER: redirect message to user
                    `get VARNAME: get variable in dialog status
                    `set VARNAME VAR: set variable in dialog status
                    `h, `help: print help
            """
        else:
            current_status["$RESPONSE"] = "Unknown command"
        return current_status

    def get_response(self, status_data, current_status, incre_state=None, **args):
        utterance = current_status["$UTTERANCE"].strip()
        if utterance[0] != "`":
            return None, 0
        return "Success", 1
