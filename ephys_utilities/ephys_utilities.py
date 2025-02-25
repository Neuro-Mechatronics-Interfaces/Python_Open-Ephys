""" Variety of python classes and functions for working with Open Ephys applications and hardware

Author: Jonathan SHulgach
Last Updated: 02/25/2025

"""

import zmq
import json
import uuid
import time
from collections import deque
import numpy as np
from threading import Thread, current_thread
from open_ephys.analysis import Session

class Event(object):
    """ Represents an event received from a ZMQ Interface plugin """
    event_types = {0: 'TIMESTAMP', 1: 'BUFFER_SIZE', 2: 'PARAMETER_CHANGE',
                   3: 'TTL', 4: 'SPIKE', 5: 'MESSAGE', 6: 'BINARY_MSG'}

    def __init__(self, _d, _data=None):
        self.type = None
        self.stream = ''
        self.sample_num = 0
        self.source_node = 0
        self.event_state = 0
        self.event_line = 0
        self.event_word = 0
        self.numBytes = 0
        self.data = b''
        self.__dict__.update(_d)
        self.timestamp = None

        # noinspection PyTypeChecker
        self.type = Event.event_types[self.type]
        if _data:
            self.data = _data
            self.numBytes = len(_data)

            dfb = np.frombuffer(self.data, dtype=np.uint8)
            self.event_line = dfb[0]

            dfb = np.frombuffer(self.data, dtype=np.uint8, offset=1)
            self.event_state = dfb[0]

            dfb = np.frombuffer(self.data, dtype=np.uint64, offset=2)
            self.event_word = dfb[0]
        if self.type == 'TIMESTAMP':
            t = np.frombuffer(self.data, dtype=np.int64)
            self.timestamp = t[0]

    def set_data(self, _data):
        """ Sets event data """
        self.data = _data
        self.numBytes = len(_data)

    def __str__(self):
        """Prints info about the event"""
        ds = self.__dict__.copy()
        del ds['data']
        return str(ds)

class Spike(object):
    """ Represents a spike event received from a ZMQ Interface plugin """
    def __init__(self, _d, _data=None):
        self.stream = ''
        self.source_node = 0
        self.electrode = 0
        self.sample_num = 0
        self.num_channels = 0
        self.num_samples = 0
        self.sorted_id = 0
        self.threshold = []

        self.__dict__.update(_d)
        self.data = _data

    def __str__(self):
        ds = self.__dict__.copy()
        del ds['data']
        return str(ds)

class OpenEphysClient(object):
    """
    Python app used to test the ZMQ Interface plugin
    """

    def __init__(self,
                 name="OpenEphysClient",
                 ip="tcp://localhost",
                 data_port=5556,
                 heartbeat_port=5557,
                 interval=0.1,
                 heartbeat_rate=2,
                 num_channels=138,
                 buffer_len=2000,
                 verbose=False,
                 ):
        self._timer = None
        self.name = name
        self.ip = ip
        self.data_port = data_port
        self.heartbeat_port = heartbeat_port
        self.interval = interval
        self.heartbeat_rate = heartbeat_rate
        self.verbose = verbose

        self.is_running = False
        self.context = zmq.Context()
        self.heartbeat_socket = None
        self.data_socket = None
        self.poller = zmq.Poller()
        self.message_num = 0
        self.socket_waits_reply = False
        self.uuid = str(uuid.uuid4())
        self.last_heartbeat_time = 0
        self.last_reply_time = time.time()
        self.prev_time = time.perf_counter()

        self.num_channels = num_channels
        self.buffer_len = buffer_len

        # Create a deque for each channel to store the data
        self.buffers = [deque(maxlen=self.buffer_len) for _ in range(self.num_channels)]

        self.initialize_sockets()
        self.thread = Thread(target=self.callback, daemon=True)
        self.thread.start()

    def initialize_sockets(self):
        """Initialize the data socket"""
        if not self.data_socket:
            ip_string = f'{self.ip}:{self.data_port}'
            print("Initializing data socket on " + ip_string)
            self.data_socket = self.context.socket(zmq.SUB)
            self.data_socket.connect(ip_string)
            self.data_socket.setsockopt(zmq.SUBSCRIBE, b'')
            self.poller.register(self.data_socket, zmq.POLLIN)

        if not self.heartbeat_socket:
            ip_string = f'{self.ip}:{self.heartbeat_port}'
            print("Initializing heartbeat socket on " + ip_string)
            self.heartbeat_socket = self.context.socket(zmq.REQ)
            self.heartbeat_socket.connect(ip_string)
            self.poller.register(self.heartbeat_socket, zmq.POLLIN)

    def send_heartbeat(self):
        """Sends heartbeat message to ZMQ Interface to indicate that the app is alive
        """
        d = {'application': self.name, 'uuid': self.uuid, 'type': 'heartbeat'}
        j_msg = json.dumps(d)
        print("Sending heartbeat...")
        self.heartbeat_socket.send(j_msg.encode('utf-8'))
        self.last_heartbeat_time = time.time()
        self.socket_waits_reply = True

    def callback(self):

        t = current_thread()
        t.alive = True
        while t.alive:
            # Periodically send heartbeats
            if (time.time() - self.last_heartbeat_time) > self.heartbeat_rate:
                if self.socket_waits_reply:
                    print("heartbeat haven't got reply, retrying...")
                    self.last_heartbeat_time += 1.
                    if (time.time() - self.last_reply_time) > 10.:
                        # reconnecting the socket as per the "lazy pirate" pattern (see the ZeroMQ guide)
                        print("connection lost, trying to reconnect")
                        self.poller.unregister(self.data_socket)
                        self.data_socket.close()
                        self.data_socket = None
                        self.poller.unregister(self.heartbeat_socket)
                        self.heartbeat_socket.close()
                        self.heartbeat_socket = None

                        self.initialize_sockets()
                        self.socket_waits_reply = False
                        self.last_reply_time = time.time()
                else:
                    if self.socket_waits_reply == True:
                        self.send_heartbeat()

            # check poller
            socks = dict(self.poller.poll(1))
            if not socks:
                continue
            if self.data_socket in socks:
                try:
                    message = self.data_socket.recv_multipart(zmq.NOBLOCK)
                except zmq.ZMQError as err:
                    print("got error: {0}".format(err))
                    break

                if message:
                    self.message_num += 1
                    if len(message) < 2:
                        print("no frames for message: ", message[0])
                        continue
                    try:
                        header = json.loads(message[1].decode('utf-8'))
                    except ValueError as e:
                        print("ValueError: ", e)
                        print(message[1])
                        continue

                    if header['message_num'] != self.message_num:
                        print("Missed a message at number", self.message_num)
                    self.message_num = header['message_num']

                    if header['type'] == 'data':
                        c = header['content']
                        num_samples = c['num_samples']
                        channel_num = c['channel_num']
                        sample_rate = c['sample_rate']

                        if self.verbose:
                            print(f"Received {num_samples} samples")
                            print(f"Channel number: {channel_num}")
                            print(f"Sample rate: {sample_rate}")

                        # Convert the raw bytes to a NumPy array of float32
                        try:
                            n_arr = np.frombuffer(message[2], dtype=np.float32)
                            n_arr = np.reshape(n_arr, num_samples)

                            if 0 <= channel_num < self.num_channels:
                                self.buffers[channel_num].extend(n_arr)
                            else:
                                print(f"Invalid channel number: {channel_num}")

                        except IndexError as e:
                            print(f"Error processing data: {e}")

                    elif header['type'] == 'event':
                        if header['data_size'] > 0:
                            event = Event(header['content'], message[2])
                        else:
                            event = Event(header['content'])
                        print(event)

                    elif header['type'] == 'spike':
                        spike = Spike(header['spike'], message[2])
                        print(spike)

                    else:
                        raise ValueError("message type unknown")
                else:
                    print("No data in message, breaking")

            elif self.heartbeat_socket in socks and self.socket_waits_reply:
                message = self.heartbeat_socket.recv()
                print(f'Heartbeat reply: {message}')
                if self.socket_waits_reply:
                    self.socket_waits_reply = False
                    self.last_reply_time = time.time()
                else:
                    print("Received reply before sending a message?")

    def get_samples(self, channel=0, n_samples=1):
        """ Returns the most recent samples up to n_samples from the specified channel """
        if 0 <= channel < self.num_channels:
            return list(self.buffers[channel]) #[-n_samples:])
        else:
            print(f"Invalid channel number: {channel}")
            return []

    def get_latest_sample_old(self):
        """ Returns the latest sample from all channels in the buffer """
        temp = []
        for i in range(self.num_channels):
            if len(self.buffers[i]) > 0:
                temp.append( np.array(self.buffers[i], dtype=np.float32) )
            else:
                temp.append(0)
        return temp

    def get_latest_sample(self):
        """Return the latest (most recent) sample from each channel's buffer."""
        latest_samples = []
        for channel_buffer in self.buffers:
            if len(channel_buffer) > 0:
                latest_samples.append(channel_buffer[-1])  # The newest item is at index -1
            else:
                # If the buffer is empty, decide how you want to handle that—maybe append None or 0
                latest_samples.append(0)
        return latest_samples

def load_file(filepath, verbose=False):
    """Load an Open Ephys file and return the data as a dictionary, matching the style of the intan .rhd files"""
    session = Session(filepath)
    #if verbose:
    #    print(session)
    recording = session.recordnodes[0].recordings[0] # Loads sample numbers, timestamps, and metadata
    #print(dir(recording))
    #print(recording.info)
    #print(dir(recording.continuous[0]))

    #continuous = recording.continuous[0]
    #print()

    #print(continuous.sample_rate)

    #print("before data dict")
    # Create and full a numpy array from the data contained in continuous where each row corresponds to the data for that channel
    data = {'amplifier_data': np.array(recording.continuous[0].samples, dtype=np.float32),
         't_amplifier': np.array(recording.continuous[0].timestamps, dtype=np.float64),
         #'amplifier_channels': recording.continuous[0].channels,
         'frequency_parameters': {'amplifier_sample_rate': recording.info['continuous'][0]['sample_rate'], 'aux_input_sample_rate': 1000},
         'info': recording.info,
         'board_adc_data': None, # Sync data
         'aux_input_data': None, # Aux data
         'continuous': recording.continuous[0],
         }

    for i in data['info']:
        print(f"{i}: {data['info'][i]}\n")

    return data, True