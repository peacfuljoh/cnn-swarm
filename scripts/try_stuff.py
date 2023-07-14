
# from ossr_utils.io_utils import save_json
# save_json('blah.json', dict(a=None))

# import platform    # For getting the operating system name
# import subprocess  # For executing a shell command
#
# def ping(host):
#     """
#     Returns True if host (str) responds to a ping request.
#     Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
#     """
#
#     # Option for the number of packets as a function of
#     param = '-n' if platform.system().lower()=='windows' else '-c'
#
#     # Building the command. Ex: "ping -c 1 google.com"
#     command = ['ping', param, '1', host, '-t', '1']
#
#     return subprocess.call(command) == 0
#
# print(ping('10.0.0.34'))

import requests
import time

url_base = 'http://30.0.0.'
url_suffix = ':48515/type'
# url_ = 'http://10.0.0.34:48515/type'

for i in range(1, 255):
    url_ = url_base + str(i) + url_suffix
    try:
        t0 = time.time()
        requests.get(url_, timeout=0.01).json()
        print(time.time() - t0)
        print('*Success for ' + url_)
    except:
        print('Timeout for ' + url_)

# from socket import *
# import time
#
# startTime = time.time()
#
# if __name__ == '__main__':
#     # target = input('Enter the host to be scanned: ')
#     target = '10.0.0.34'
#     t_IP = gethostbyname(target)
#     print('Starting scan on host: ', t_IP)
#
#     for i in range(1, 255):
#         s = socket(AF_INET, SOCK_STREAM)
#
#         conn = s.connect_ex((t_IP, i))
#         if (conn == 0):
#             print('Port %d: OPEN' % (i,))
#         s.close()
# print('Time taken:', time.time() - startTime)

# import os
# import platform
#
# from datetime import datetime
#
# # net = input("Enter the Network Address: ")
# net = '10.0.0.34'
# net1 = net.split('.')
# a = '.'
#
# net2 = net1[0] + a + net1[1] + a + net1[2] + a
# # st1 = int(input("Enter the Starting Number: "))
# # en1 = int(input("Enter the Last Number: "))
# st1 = 1
# en1 = 10
# en1 = en1 + 1
# # oper = platform.system()
#
# # if (oper == "Windows"):
# #     ping1 = "ping -n 1 "
# # elif (oper == "Linux"):
# ping1 = "ping -c 1 "
# # else:
# #     ping1 = "ping -c 1 "
# t1 = datetime.now()
# print("Scanning in Progress:")
#
# for ip in range(st1, en1):
#     addr = net2 + str(ip)
#     comm = ping1 + addr
#     response = os.popen(comm)
#
#     for line in response.readlines():
#         if (line.count("TTL")):
#             break
#         if (line.count("TTL")):
#             print(addr, "--> Live")
#
# t2 = datetime.now()
# total = t2 - t1
# print("Scanning completed in: ", total)
