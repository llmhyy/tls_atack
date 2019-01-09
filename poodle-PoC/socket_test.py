import socket
import socketserver
import threading

class MyTCPHandler(socketserver.BaseRequestHandler):
	"""
	The request handler class for our server.

	It is instantiated once per connection to the server, and must
	override the handle() method to implement communication to the
	client.
	"""

	def handle(self):
		# self.request is the TCP socket connected to the client
		self.data = self.request.recv(1024).strip()
		print("{} wrote:".format(self.client_address[0]))
		print(self.data)
		time.sleep(5)
		# just send back the same data, but upper-cased
		self.request.sendall(self.data.upper())


httpd = socketserver.TCPServer(("127.0.0.1", 4444), MyTCPHandler)
proxy = threading.Thread(target=httpd.serve_forever)
proxy.daemon=True
proxy.start()

print("Socket launched")

while True:
	input_u = input("> ")
	if input_u == "exit":
		break
