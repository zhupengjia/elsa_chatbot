#!/usr/bin/env python
import nbxmpp
from gi.repository import GObject as gobject

class XMPP:
    def __init__(self, jid, password):
        self.jid = nbxmpp.protocol.JID(jid)
        self.password = password
        self.sm = nbxmpp.Smacks(self) # Stream Management
        self.client_cert = None
        self.connect()

   def on_auth(self, con, auth):
        if not auth:
            print('could not authenticate!')
            sys.exit()
        print('authenticated using ' + auth)
        self.send_message(to_jid, text)

    def on_connected(self, con, con_type):
        print('connected with ' + con_type)
        auth = self.client.auth(self.jid.getNode(), self.password, resource=self.jid.getResource(), sasl=1, on_auth=self.on_auth)

    def get_password(self, cb, mech):
        cb(self.password)

    def on_connection_failed(self):
        print('could not connect!')

    def _event_dispatcher(self, realm, event, data):
        pass

    def connect(self):
        idle_queue = nbxmpp.idlequeue.get_idlequeue()
        self.client = nbxmpp.NonBlockingClient(self.jid.getDomain(), idle_queue, caller=self)
        self.con = self.client.connect(self.on_connected, self.on_connection_failed, secure_tuple=('tls', '', '', None, None))

    def interact(self, query, to_jid):
        id_ = self.client.send(nbxmpp.protocol.Message(to_jid, query, typ='chat'))
        print('sent message with id ' + id_)
        

    def quit(self):
        self.disconnect()
        ml.quit()

    def disconnect(self):
        self.client.start_disconnect()

    def run(self):
        ml = gobject.MainLoop()
        ml.run()


