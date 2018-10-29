import os
import subprocess

result = subprocess.run(["./testssl.sh", "-O", "157.240.13.35:443"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
# testing with facebook
y = result.stdout
x = result.stderr
yh = y.decode("utf-8")
poodle_string = "POODLE, SSL\x1b[m (CVE-2014-3566) "
NON_valid = "Unable to open a socket"
data_set = []
working_set = []

def scan():
    for ip1 in range(0, 255):
        print(ip1)
        for ip2 in range(0, 256):
            print(ip2)
            for ip3 in range(0, 259):
                print(ip3)
                for ip4 in range(0, 255):

                    ip = str(ip1) + "." + str(ip2) + "." + str(ip3) + "." + str(ip4) + ":443"
                    print(ip)
                    scan = subprocess.run(["./testssl.sh", "-O", ip],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
                    print(scan.stdout)
                    scan_string = scan.stdout.decode("utf-8")

                    if NON_valid in scan_string:
                        print(scan_string)
                        this_ip = {ip: "invalid ip"}
                        print(this_ip)
                    elif poodle_string in scan_string:
                        print("valid IP")
                        string_index = scan_string(poodle_string)
                        if "not vulnerable (OK)" in scan_string:
                            self.ip = {ip, "not vulnerable (OK)"}
                            data_set.append(this_ip)
                        else:
                            this_ip = {ip: scan_string}
                            working_set.append(this_ip)
                            print(working_set)
                            with open("data_set.txt", "a")  as f:
                                f.write(str(this_ip) + "\n")


scan()
