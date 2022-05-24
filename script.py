import os
from time import sleep
import sys
import logging
import colorlog
from subprocess import Popen, PIPE, STDOUT
import subprocess 
import time
import signal 

cmd= "/home/brai-agx-1/build-br-peak-opencv-Desktop-Debug/br-peak-opencv"
target_process= "br-peak-opencv"

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('my_log_info.log')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)r', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(colorlog.ColoredFormatter('%(log_color)s [%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S'))
logger.addHandler(fh)
logger.addHandler(sh)
error = 0
seconds = (60*59)+30
#seconds =30

import psutil
import threading

def on_timeout(proc, status_dict):
	"""Kill process on timeout and note as status_dict['timeout']=True"""
	# a container used to pass status back to calling thread
	status_dict['timeout'] = True
	logger.warning("App Timeout")	
	try:
		proc.wait(timeout=3)
	except subprocess.TimeoutExpired:
		kill(proc.pid)

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()



succ = 0
fail = 0

while True:
	subproc = subprocess.Popen([cmd], stdout = subprocess.PIPE, stderr=subprocess.STDOUT, shell=True) 
	status_dict = {'timeout':False}
	# Start timer thread 
	timer = threading.Timer(seconds, on_timeout, (subproc, status_dict))
	timer.start()
	# Output pipeline
	pipe = subproc.stdout
	exit_flag = 0
	for line in iter(pipe.readline,b''):
		line = line.decode(sys.stdout.encoding)
		lineLow = line.lower()
		if "error" in lineLow:

			if "QJsonValue" in line:
				logger.error(line)

			else:
				logger.error(line)
				exit_flag = 1
				logger.warning("Restarting App")
				try:
				    subproc.wait(timeout=3)
				except subprocess.TimeoutExpired:
				    kill(subproc.pid)
		elif "Buffer incomplete"  in line:
			#logger.warning(line)
			y=1
		elif "exception"  in lineLow:
			logger.warning(line)
		else:	
			logger.info(line)

	# End process
	subproc.wait()
    # in case we didn't hit timeout
	timer.cancel()
	logger.warning(status_dict)
	
	# Check for timeout
	if not status_dict['timeout'] and subproc.returncode == 100:
		succ += 1 # If returned 100 then success
		logger.warning("App Timeout")	
		try:
			subproc.wait(timeout=3)
		except subprocess.TimeoutExpired:
			kill(subproc.pid)
	else:
		fail += 1 # Else Failed
		#break # Break on failure		
	logger.warning("Clearning App Memory")	
	time.sleep(15)





